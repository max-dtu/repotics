import argparse
import base64
import json
import logging
import os
import socket
import time
import urllib.request
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image
import requests

from camera.frame_reader import FrameReader
from robot.commands import Command, AVAILABLE_COMMANDS

ALLOWED_COMMANDS = set(c.value for c in AVAILABLE_COMMANDS)
_cmds_str = ", ".join(c.value for c in AVAILABLE_COMMANDS)

CLASSES_OF_INTEREST = set(["ball", "car"])
_CLASSES_OF_INTEREST_str = ",".join(CLASSES_OF_INTEREST)

DEFAULT_PROMPT = (
    f'You control a robot by choosing the best available command from [{_cmds_str}] based on the current task and the camera frame. '
    f'Current task: {{task}} See the camera frame, detect {_CLASSES_OF_INTEREST_str}; evaluate; and then respond. '
    'If unsure which command to choose, choose left or right command randomly. '
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


def frame_to_image(frame, width: int, height: int) -> Image.Image:
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def save_latest_frame(image: Image.Image, path: str | None) -> None:
    if not path:
        return
    image.save(path)


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


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


def query_gemini(api_key: str, model: str, base64_image: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    response = requests.post(url, headers=headers, json=payload, timeout=10.0)
    response.raise_for_status()
    result = response.json()
    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return text
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response structure from Gemini API: {result}") from e


def query_openai(api_key: str, model: str, base64_image: str, prompt: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload, timeout=10.0)
    response.raise_for_status()
    result = response.json()
    try:
        text = result["choices"][0]["message"]["content"]
        return text
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response structure from OpenAI API: {result}") from e


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


def run(args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Validate API key based on selected model
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    is_openai = "gpt" in args.model.lower()
    if is_openai and not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required to run OpenAI models.")
    elif not is_openai and not gemini_key:
        raise ValueError("GEMINI_API_KEY environment variable is required to run Gemini models.")

    camera = FrameReader(index=args.camera_index)
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
        # Warmup delay for camera exposure auto-adjustment
        time.sleep(1.0)
        
        while args.max_steps is None or steps < args.max_steps:
            started_at = time.monotonic()
            frame = camera.get_frame(timeout=args.timeout)
            image = frame_to_image(frame, args.width, args.height)
            save_latest_frame(image, args.save_latest_frame)
            
            base64_image = image_to_base64(image)
            prompt = build_task_prompt(args.prompt, task)
            
            logger.info("Model string input (prompt):\n%s", prompt)
            
            # Query VLM API
            try:
                if is_openai:
                    raw_output = query_openai(openai_key, args.model, base64_image, prompt)
                else:
                    raw_output = query_gemini(gemini_key, args.model, base64_image, prompt)
            except Exception as e:
                logger.error("API request failed: %s", e)
                steps += 1
                time.sleep(max(0.0, interval - (time.monotonic() - started_at)))
                continue

            decision = parse_decision(raw_output)
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
    parser = argparse.ArgumentParser(description="Drive an EV3 robot with cloud model decisions (OpenAI or Gemini).")
    parser.add_argument("--host", default="10.45.151.18", help="EV3 TCP host, or ntfy:<topic> for remote mode.")
    parser.add_argument("--port", type=int, default=9999, help="EV3 TCP port.")
    parser.add_argument(
        "--model", 
        default="gemini-2.0-flash", 
        help="Model identifier (e.g. gemini-2.0-flash, gemini-1.5-flash, gpt-4o, gpt-4o-mini)."
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
        help="Ask for a task at startup and again whenever model reports task_done=true.",
    )
    parser.add_argument(
        "--save-latest-frame",
        default="cloud_latest_frame.jpg",
        help="Path to overwrite with the latest resized frame sent to model. Use '' to disable.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Decision prompt sent with each frame.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
