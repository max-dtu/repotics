import argparse
import json
import logging
import os
import socket
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image

from camera import Camera
from robot import Command


DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_PROMPT = (
    "You control a robot. Current task: {task}. "
    "Goal: satisfy the task using the camera frame. "
    "Choose exactly one command: left, right, forward, backward. "
    "If unsure, rotate left or right. "
    "Also decide whether the task is already complete. "
    "Reply only with compact JSON in this exact shape: "
    '{"command":"left|right|forward|backward","task_done":true|false,'
    '"car_visible":true|false,"ball_visible":true|false,'
    '"target":"car|ball|unknown","reason":"short reason"}.'
)
ALLOWED_COMMANDS = {
    Command.LEFT.value,
    Command.RIGHT.value,
    Command.FORWARD.value,
    Command.BACKWARD.value,
}

logger = logging.getLogger(__name__)


@dataclass
class VisionDecision:
    command: str | None
    task_done: bool = False
    car_visible: bool | None = None
    ball_visible: bool | None = None
    target: str = "unknown"
    reason: str = ""
    raw_output: str = ""


class EV3TcpClient:
    def __init__(self, host: str, port: int, timeout: float = 3.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None

    def send(self, command: str) -> bool:
        payload = f"{command}\n".encode("utf-8")
        for _ in range(2):
            try:
                if self._sock is None:
                    self._sock = socket.create_connection(
                        (self.host, self.port),
                        timeout=self.timeout,
                    )
                    logger.info("Connected to EV3 at %s:%s", self.host, self.port)
                self._sock.sendall(payload)
                self._sock.recv(16)
                return True
            except OSError as exc:
                logger.warning("EV3 send failed (%s); reconnecting", exc)
                self.close()
        return False


class EV3NtfyClient:
    def __init__(self, topic: str, timeout: float = 3.0) -> None:
        self.topic = topic
        self.timeout = timeout
        self.url = f"https://ntfy.sh/{topic}"

    def close(self) -> None:
        return

    def send(self, command: str) -> bool:
        try:
            request = urllib.request.Request(
                self.url,
                data=command.encode("utf-8"),
                headers={"User-Agent": "Mozilla/5.0"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
            return True
        except Exception as exc:
            logger.warning("ntfy send failed (%s)", exc)
            return False


def build_ev3_client(host: str, port: int, timeout: float):
    if host.startswith("ntfy:") or host.startswith("dweet:"):
        topic = host.split(":", 1)[1].strip()
        if not topic:
            raise ValueError("Remote host must include a topic, for example ntfy:kals-ev3")
        logger.info("Using ntfy remote mode topic: %s", topic)
        return EV3NtfyClient(topic, timeout=timeout)
    return EV3TcpClient(host, port, timeout=timeout)


def configure_huggingface_cache() -> None:
    if "HF_HOME" in os.environ:
        return
    cache_dir = Path(__file__).resolve().parent / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)


def frame_to_image(frame, width: int, height: int) -> Image.Image:
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def save_latest_frame(image: Image.Image, path: str | None) -> None:
    if not path:
        return
    image.save(path)


def parse_command(text: str) -> str | None:
    cleaned = text.strip().lower().strip(".,:;!?\"'`")
    if cleaned in ALLOWED_COMMANDS:
        return cleaned

    first_word = cleaned.split(maxsplit=1)[0] if cleaned else ""
    if first_word in ALLOWED_COMMANDS:
        logger.warning("Model produced extra text %r; using first word %r", text, first_word)
        return first_word

    logger.warning("Ignoring invalid model output: %r", text)
    return None


def _extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_decision(text: str) -> VisionDecision:
    data = _extract_json_object(text)
    if data is not None:
        command = parse_command(str(data.get("command", "")))
        return VisionDecision(
            command=command,
            task_done=data.get("task_done") if isinstance(data.get("task_done"), bool) else False,
            car_visible=data.get("car_visible") if isinstance(data.get("car_visible"), bool) else None,
            ball_visible=data.get("ball_visible") if isinstance(data.get("ball_visible"), bool) else None,
            target=str(data.get("target", "unknown"))[:40],
            reason=str(data.get("reason", ""))[:240],
            raw_output=text,
        )

    return VisionDecision(command=parse_command(text), raw_output=text)


def model_input_device(model):
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def build_task_prompt(prompt_template: str, task: str) -> str:
    if "{task}" in prompt_template:
        return prompt_template.replace("{task}", task)
    return f"{prompt_template}\nCurrent task: {task}"


def ask_for_task(default_task: str | None = None) -> str | None:
    suffix = f" [{default_task}]" if default_task else ""
    try:
        task = input(f"Robot task{suffix}> ").strip()
    except EOFError:
        return default_task
    if task:
        return task
    return default_task


def choose_command(model, processor, image: Image.Image, prompt: str) -> VisionDecision:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "resized_width": image.width,
                    "resized_height": image.height,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model_input_device(model))

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)

    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_decision(output)


def resolve_torch_dtype(dtype: str, device: str):
    import torch

    if dtype == "auto":
        if device == "mps" or (device == "auto" and torch.backends.mps.is_available()):
            return torch.float16
        if device == "cpu" or not torch.cuda.is_available():
            return torch.float32
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def resolve_device_map(device: str):
    if device == "auto":
        return "auto"
    if device == "cpu":
        return {"": "cpu"}
    if device == "mps":
        return {"": "mps"}
    raise ValueError(f"Unsupported device: {device}")


def load_model(model_name: str, dtype: str, device: str, local_files_only: bool = False):
    configure_huggingface_cache()

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    logger.info("Loading %s", model_name)
    torch_dtype = resolve_torch_dtype(dtype, device)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=resolve_device_map(device),
        local_files_only=local_files_only,
    )
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    processor = AutoProcessor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    return model, processor


def run(args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    model, processor = load_model(
        args.model,
        dtype=args.torch_dtype,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    camera = Camera(index=args.camera_index)
    ev3 = build_ev3_client(args.host, args.port, timeout=args.timeout)
    interval = 1.0 / args.hz
    steps = 0
    task = args.task
    if args.ask_task:
        task = ask_for_task(task)
    if not task:
        logger.info("No task provided; exiting.")
        return
    logger.info("Active task: %s", task)

    try:
        camera.open()
        while args.max_steps is None or steps < args.max_steps:
            started_at = time.monotonic()
            frame = camera.capture(timeout=args.timeout)
            image = frame_to_image(frame, args.width, args.height)
            save_latest_frame(image, args.save_latest_frame)
            decision = choose_command(model, processor, image, build_task_prompt(args.prompt, task))
            command = decision.command

            logger.info(
                "Vision decision: task=%r done=%s command=%s car_visible=%s ball_visible=%s target=%s reason=%s",
                task,
                decision.task_done,
                command,
                decision.car_visible,
                decision.ball_visible,
                decision.target,
                decision.reason or "<none>",
            )
            logger.info("Model raw output: %s", decision.raw_output.strip())
            if args.save_latest_frame:
                logger.info("Model frame: %s (%dx%d)", args.save_latest_frame, image.width, image.height)

            if decision.task_done:
                logger.info("Task complete: %s", task)
                if not args.ask_task:
                    break
                task = ask_for_task()
                if not task:
                    logger.info("No next task provided; exiting.")
                    break
                logger.info("Active task: %s", task)
                continue

            if command is not None:
                if args.dry_run:
                    logger.info("Dry run: %s", command)
                elif ev3.send(command):
                    logger.info("Sent: %s", command)
                else:
                    logger.error("Failed to send: %s", command)

            steps += 1
            elapsed = time.monotonic() - started_at
            time.sleep(max(0.0, interval - elapsed))
    finally:
        ev3.close()
        camera.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drive an EV3 robot with Qwen2-VL camera decisions.")
    parser.add_argument("--host", default="10.45.151.18", help="EV3 TCP host, or ntfy:<topic> for remote mode.")
    parser.add_argument("--port", type=int, default=9999, help="EV3 TCP port.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name or local path.")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "float32", "bfloat16"),
        default="auto",
        help="Model dtype. auto uses float16 on MPS, float32 on CPU, and Transformers auto on CUDA.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Device placement. Use cpu for the most reliable Intel Mac path.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only from the local Hugging Face cache.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--hz", type=float, default=1.0, help="Decision frequency. Keep this around 1-2 Hz.")
    parser.add_argument("--width", type=int, default=320, help="Image width passed to the VLM.")
    parser.add_argument("--height", type=int, default=240, help="Image height passed to the VLM.")
    parser.add_argument("--timeout", type=float, default=3.0, help="Camera and socket timeout in seconds.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many loop iterations.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without sending to EV3.")
    parser.add_argument(
        "--task",
        default="move toward the ball",
        help="Initial robot task. Used as the default when interactive task prompting is enabled.",
    )
    parser.add_argument(
        "--ask-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask for a task at startup and again whenever Qwen reports task_done=true.",
    )
    parser.add_argument(
        "--save-latest-frame",
        default="qwen_latest_frame.jpg",
        help="Path to overwrite with the latest resized frame sent to Qwen. Use '' to disable.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Decision prompt sent with each frame.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
