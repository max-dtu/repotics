from functorch.experimental import control_flow
import argparse
import json
import logging
import os
import socket
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Enable high watermark ratio bypass to allow MPS to use system RAM paging on macOS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import cv2
from PIL import Image

from camera import Camera
from robot import Command


AVAILABLE_COMMANDS = [
    Command.LEFT.value,
    Command.RIGHT.value,
    Command.FORWARD.value,
    Command.BACKWARD.value,
    Command.OPEN_GRIPPER.value,
    Command.CLOSE_GRIPPER.value,
]
ALLOWED_COMMANDS = set(AVAILABLE_COMMANDS)

_cmds_str = ", ".join(AVAILABLE_COMMANDS)

CLASSES_OF_INTEREST = set(["ball","car"])
_CLASSES_OF_INTEREST_str = ",".join(CLASSES_OF_INTEREST)

# Available choices e.g.: "InternRobotics/RoboInter-VLM", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2-VL-2B-Instruct, Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_PROMPT = (
    f'You control a robot by choosing the best available command from [{_cmds_str}] based on the current task and the camera frame. Current task: {{task}} See the camera frame, detect {_CLASSES_OF_INTEREST_str}; evaluate; and then respond. If unsure which command to choose, choose left or right command randomly. '
    'Also decide whether the task is already complete. If the task is complete, set "task_done" to true. '
    'Identify which of the visible objects of interest is closest to the robot. '
    'Reply only with compact JSON in this exact shape: {"command":"...","task_done":true|false,"visible_objects_of_interest":[...],"object_closest_to_robot":"..."}'
)

logger = logging.getLogger(__name__)


@dataclass
class VisionDecision:
    command: str | None
    task_done: bool = False
    visible_objects_of_interest: set[str] | None = None
    objects_of_interest_closest: str | None = None
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


def wrap_stand_out(label: str, text: str) -> str:
    color_border = "\033[1;35m" # Bold Magenta
    color_content = "\033[1;32m" # Bold Green
    reset = "\033[0m"
    
    lines = text.strip().split("\n")
    width = max(len(line) for line in lines) if lines else 0
    width = max(width, len(label))
    
    border = "═" * (width + 4)
    
    result = []
    result.append(f"{color_border}╔{border}╗{reset}")
    result.append(f"{color_border}║  {color_content}{label.center(width)}{color_border}  ║{reset}")
    result.append(f"{color_border}╠{border}╣{reset}")
    for line in lines:
        result.append(f"{color_border}║  {color_content}{line.ljust(width)}{color_border}  ║{reset}")
    result.append(f"{color_border}╚{border}╝{reset}")
    
    return "\n" + "\n".join(result)


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
        
        visible = data.get("visible_objects_of_interest")
        visible_set = None
        if isinstance(visible, str):
            visible_set = {x.strip() for x in visible.replace("|", ",").split(",") if x.strip()}
        elif isinstance(visible, list):
            visible_set = {str(x).strip() for x in visible if str(x).strip()}
            
        closest = data.get("object_closest_to_robot")
        closest_str = str(closest).strip() if closest is not None else None

        return VisionDecision(
            command=command,
            task_done=data.get("task_done") if isinstance(data.get("task_done"), bool) else False,
            visible_objects_of_interest=visible_set,
            objects_of_interest_closest=closest_str,
            raw_output=text,
        )

    return VisionDecision(command=parse_command(text), raw_output=text)


def model_input_device(model):
    import torch

    if hasattr(model, "model"):
        try:
            return next(model.model.parameters()).device
        except StopIteration:
            pass
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
    
    logger.info("Model string input (prompt):\n%s", prompt)
    logger.info("Model string input (chat_text):\n%s", chat_text)

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
    
    logger.info(wrap_stand_out("Model string output", output))
    
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
        # Offload the visual transformer block (Conv3D) to CPU to avoid MPS not supported crash,
        # and map everything else to mps.
        return {
            "visual": "cpu",
            "": "mps"
        }
    raise ValueError(f"Unsupported device: {device}")


def load_model(model_name: str, dtype: str, device: str, local_files_only: bool = False):
    configure_huggingface_cache()

    import torch
    # limit threads to prevent system starvation on CPU
    if device in ("cpu", "mps") or (device == "auto" and not torch.cuda.is_available()):
        num_cores = os.cpu_count() or 4
        torch_threads = max(1, num_cores - 2)
        torch.set_num_threads(torch_threads)
        logger.info("Set torch CPU threads to %d (system has %d cores) to prevent thread starvation", torch_threads, num_cores)

    from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig

    logger.info("Loading %s", model_name)
    torch_dtype = resolve_torch_dtype(dtype, device)
    config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    if hasattr(config, "text_config") and isinstance(config.text_config, dict):
        delattr(config, "text_config")

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map=resolve_device_map(device),
        local_files_only=local_files_only,
    )

    # Workaround for MPS not supporting Conv3D & Float16 LayerNorm on CPU:
    # If the device is MPS, we configure a hybrid CPU-Float32 and MPS-Float16 device map.
    if device == "mps":
        # 1. Remove the default accelerate hook from visual submodule to prevent it from moving back to MPS
        if hasattr(model.visual, "_hf_hook"):
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(model.visual, recurse=True)

        # 2. Move the visual submodule to CPU in Float32 to bypass PyTorch CPU LayerNorm Float16 constraint
        model.visual.to(device="cpu", dtype=torch.float32)

        # 3. Override model.device property so model.generate knows the language model is running on MPS
        type(model).device = property(lambda self: next(self.model.parameters()).device)

        # 4. Register pre-forward hook to cast inputs to Float32 on CPU when entering visual submodule
        def move_inputs_to_cpu_float32(module, args, kwargs):
            new_args = tuple(x.to("cpu", dtype=torch.float32) if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in args)
            new_kwargs = {k: v.to("cpu", dtype=torch.float32) if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in kwargs.items()}
            return new_args, new_kwargs

        # 5. Register forward hook to cast outputs to Float16 and move them back to MPS when leaving visual submodule
        def move_outputs_to_mps_float16(module, input, output):
            if isinstance(output, torch.Tensor):
                return output.to("mps", dtype=torch.float16)
            elif isinstance(output, tuple):
                return tuple(x.to("mps", dtype=torch.float16) if isinstance(x, torch.Tensor) else x for x in output)
            return output

        model.visual.register_forward_pre_hook(move_inputs_to_cpu_float32, with_kwargs=True)
        model.visual.register_forward_hook(move_outputs_to_mps_float16)

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
                "Vision decision: task=%r done=%s command=%s objects_visible=%s closest_object=%s",
                task,
                decision.task_done,
                command,
                decision.visible_objects_of_interest,
                decision.objects_of_interest_closest or "<none>",
            )
            logger.info(wrap_stand_out("Model raw output", decision.raw_output))
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
    parser = argparse.ArgumentParser(description="Drive an EV3 robot with model camera decisions.")
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
