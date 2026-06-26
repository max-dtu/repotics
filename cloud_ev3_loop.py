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
    "You control a robot car on a 2D field. You are provided with:\n"
    "1. The Active Task: {task}\n"
    "2. The Last Executed Command: {last_command} (If this is the first step, the last command is 'none')\n"
    "3. Current Gripper State: {gripper_state} (either 'open' or 'closed')\n"
    "4. Previous Frame (before the command was executed)\n"
    "5. Current Frame (after the command was executed)\n\n"
    "The front of the robot is distinguished by a gripper/fork (forklift structure) attached to it. The robot's primary objective is to align its front (gripper) to directly face the ball before moving towards it.\n"
    "First, locate the robot car and the ball in both frames. If the last command was not 'none', compare the Previous Frame and Current Frame to evaluate if the command improved, degraded, or did not change progress towards the task (e.g. by decreasing or increasing the 2D distance or improving the alignment between the front gripper and the ball):\n"
    "- If the robot is STUCK (no movement/difference between the frames) or progress DEGRADED (the robot moved further away from the ball, or turned further away from facing the ball), you MUST choose the opposite/undo command of the last command to recover (opposite of forward is backward, backward is forward, left is right, right is left, open_gripper is close_gripper, close_gripper is open_gripper).\n"
    "- If the robot successfully made progress (moved closer to the ball, or rotated closer to facing the ball, even if not fully aligned yet), do NOT undo it; continue with the next best command to make further progress.\n"
    f"Otherwise, choose the best command from [{_cmds_str}] to rotate (left/right) or move (forward/backward) so the front gripper faces the ball.\n\n"
    "Identify which of the visible objects of interest is closest to the robot. "
    "Reply only with compact JSON in this exact shape: "
    '{"evaluation":"improved|degraded|no_change","command":"...","task_done":true|false,"visible_objects_of_interest":[...],"object_closest_to_robot":"...","gripper_identified":true|false,"gripper_orientation":"describe where the gripper is pointing relative to the ball"}'
)

logger = logging.getLogger(__name__)


@dataclass
class VisionDecision:
    command: str | None
    task_done: bool = False
    visible_objects_of_interest: set[str] | None = None
    objects_of_interest_closest: str | None = None
    evaluation: str | None = None
    gripper_identified: bool = False
    gripper_orientation: str | None = None
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

        # Parse gripper fields
        gripper_id = data.get("gripper_identified")
        gripper_identified = bool(gripper_id) if gripper_id is not None else False
        gripper_orientation = data.get("gripper_orientation")
        gripper_orientation_str = str(gripper_orientation).strip() if gripper_orientation is not None else None

        return VisionDecision(
            command=command,
            task_done=data.get("task_done") if isinstance(data.get("task_done"), bool) else False,
            visible_objects_of_interest=visible_set,
            objects_of_interest_closest=closest_str,
            evaluation=data.get("evaluation"),
            gripper_identified=gripper_identified,
            gripper_orientation=gripper_orientation_str,
            raw_output=text,
        )

    return VisionDecision(command=parse_command(text), raw_output=text)


def query_gemini(api_key: str, model: str, prev_base64_image: str, curr_base64_image: str, prompt: str, api_timeout: float = 60.0) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Previous Frame (before executing the last command):"},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": prev_base64_image
                        }
                    },
                    {"text": "Current Frame (after executing the last command):"},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": curr_base64_image
                        }
                    },
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    response = requests.post(url, headers=headers, json=payload, timeout=api_timeout)
    response.raise_for_status()
    result = response.json()
    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return text
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response structure from Gemini API: {result}") from e


def query_openai(api_key: str, model: str, prev_base64_image: str, curr_base64_image: str, prompt: str, api_base: str | None = None, api_timeout: float = 60.0) -> str:
    base_url = api_base.rstrip("/") if api_base else "https://api.openai.com/v1"
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Previous Frame (before executing the last command):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{prev_base64_image}"
                        }
                    },
                    {"type": "text", "text": "Current Frame (after executing the last command):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{curr_base64_image}"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload, timeout=api_timeout)
    response.raise_for_status()
    result = response.json()
    try:
        text = result["choices"][0]["message"]["content"]
        return text
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response structure from OpenAI API: {result}") from e


def validate_decision(
    decision: VisionDecision,
    last_command: str | None,
    gripper_open: bool
) -> str | None:
    # Level 1 check: Was JSON parsed correctly?
    if not decision.command:
        return "Error: Could not parse command from response. Ensure you choose a command from the allowed list."

    # Level 2 check: Gripper physical state constraints
    if decision.command == "open_gripper" and gripper_open:
        return "Error: Gripper is already open. You cannot open it again. Please choose a different command (e.g., forward, backward, left, right, or close_gripper)."
    if decision.command == "close_gripper" and not gripper_open:
        return "Error: Gripper is already closed. You cannot close it again. Please choose a different command (e.g., forward, backward, left, right, or open_gripper)."

    # Level 3 check: Logical consistency of evaluation vs command
    if decision.evaluation == "degraded" and last_command is not None and last_command != "none":
        if decision.command == last_command:
            opposite_cmd = None
            if last_command == "forward": opposite_cmd = "backward"
            elif last_command == "backward": opposite_cmd = "forward"
            elif last_command == "left": opposite_cmd = "right"
            elif last_command == "right": opposite_cmd = "left"
            elif last_command == "open_gripper": opposite_cmd = "close_gripper"
            elif last_command == "close_gripper": opposite_cmd = "open_gripper"
            
            suggestion = f" (e.g., '{opposite_cmd}')" if opposite_cmd else ""
            return f"Error: You evaluated that progress 'degraded' after executing '{last_command}', but you are trying to execute the exact same command again. This will cause an oscillation or get you further stuck. You must select the opposite/undo command{suggestion} or a different steering/correction command."

    if decision.evaluation == "improved" and last_command is not None and last_command != "none":
        # Check if the model is trying to undo a successful command
        opposite_cmd = None
        if last_command == "forward": opposite_cmd = "backward"
        elif last_command == "backward": opposite_cmd = "forward"
        elif last_command == "left": opposite_cmd = "right"
        elif last_command == "right": opposite_cmd = "left"
        
        if decision.command == opposite_cmd:
            return f"Error: You evaluated that progress 'improved' after '{last_command}', but you are now choosing to undo it with '{decision.command}'. If progress was successful, you should build on it rather than reversing. Please choose a command that continues progress."

    return None


def build_task_prompt(prompt_template: str, task: str, last_command: str = "none", gripper_open: bool = True) -> str:
    prompt = prompt_template
    if "{task}" in prompt:
        prompt = prompt.replace("{task}", task)
    else:
        prompt = f"{prompt}\nCurrent task: {task}"
        
    if "{last_command}" in prompt:
        prompt = prompt.replace("{last_command}", last_command)
    else:
        prompt = f"{prompt}\nLast command executed: {last_command}"
        
    gripper_state_str = "open" if gripper_open else "closed"
    if "{gripper_state}" in prompt:
        prompt = prompt.replace("{gripper_state}", gripper_state_str)
    else:
        prompt = f"{prompt}\nCurrent gripper state: {gripper_state_str}"
        
    return prompt


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

    is_openai = False
    if args.api_type == "openai":
        is_openai = True
    elif args.api_type == "gemini":
        is_openai = False
    else:
        is_openai = "gpt" in args.model.lower() or (args.api_base is not None and "gemini" not in args.api_base)

    # Validate API key only if using default cloud endpoints
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not args.api_base:
        if is_openai and not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required to run cloud OpenAI models.")
        elif not is_openai and not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is required to run cloud Gemini models.")

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
        
        prev_base64_image = None
        last_command = "none"
        gripper_open = True
        
        while args.max_steps is None or steps < args.max_steps:
            started_at = time.monotonic()
            frame = camera.get_frame(timeout=args.timeout)
            image = frame_to_image(frame, args.width, args.height)
            save_latest_frame(image, args.save_latest_frame)
            
            base64_image = image_to_base64(image)
            
            # First frame scenario
            if prev_base64_image is None:
                prev_base64_image = base64_image
                
            error_msg = None
            retry_count = 0
            max_retries = 3
            decision = None
            
            while retry_count < max_retries:
                prompt = build_task_prompt(args.prompt, task, last_command, gripper_open)
                if error_msg:
                    # Append compiler feedback to prompt
                    prompt = (
                        f"{prompt}\n\n"
                        f"### HARNESS FEEDBACK (PREVIOUS ATTEMPT FAILED):\n"
                        f"Your previous response returned the following error:\n"
                        f"{error_msg}\n"
                        f"Please correct your reasoning and select a valid command from the allowed list."
                    )
                
                logger.info("Model string input (prompt, attempt %d/%d):\n%s", retry_count + 1, max_retries, prompt)
                
                # Query VLM API
                try:
                    if is_openai:
                        raw_output = query_openai(openai_key, args.model, prev_base64_image, base64_image, prompt, api_base=args.api_base, api_timeout=args.api_timeout)
                    else:
                        raw_output = query_gemini(gemini_key, args.model, prev_base64_image, base64_image, prompt, api_timeout=args.api_timeout)
                except requests.exceptions.ConnectionError as e:
                    logger.critical("Could not connect to the API server at %s. The server may have crashed or is not running. Please check llama-server.", args.api_base or "default endpoint")
                    raise SystemExit("API Connection Error: Ensure llama-server is running.") from e
                except Exception as e:
                    logger.error("API request failed: %s (retrying without appending as model prompt feedback)", e)
                    retry_count += 1
                    continue

                decision = parse_decision(raw_output)
                
                # Validate decision
                error_msg = validate_decision(decision, last_command, gripper_open)
                if error_msg is None:
                    # Valid decision! Break loop.
                    break
                
                logger.warning("Harness rejected VLM output: %s", error_msg)
                retry_count += 1

            # If after max retries we still don't have a valid decision, skip this step
            if decision is None or error_msg is not None:
                logger.error("Harness failed to get a valid decision after %d attempts. Skipping step.", max_retries)
                steps += 1
                elapsed = time.monotonic() - started_at
                time.sleep(max(0.0, interval - elapsed))
                continue

            command = decision.command

            logger.info(
                "Vision decision: task=%r done=%s command=%s objects_visible=%s closest_object=%s evaluation=%s gripper_detected=%s orientation=%s",
                task,
                decision.task_done,
                command,
                decision.visible_objects_of_interest,
                decision.objects_of_interest_closest or "<none>",
                decision.evaluation or "<none>",
                decision.gripper_identified,
                decision.gripper_orientation or "<none>",
            )
            logger.info(wrap_stand_out("Model raw output", decision.raw_output))
            if args.save_latest_frame:
                logger.info("Model frame: %s (%dx%d)", args.save_latest_frame, image.width, image.height)

            if decision.task_done:
                logger.info("Task complete: %s", task)
                prev_base64_image = None
                last_command = "none"
                gripper_open = True
                if not args.ask_task:
                    break
                task = ask_for_task()
                if not task:
                    logger.info("No next task provided; exiting.")
                    break
                logger.info("Active task: %s", task)
                continue

            if command is not None:
                sent_success = False
                if args.dry_run:
                    logger.info("Dry run: %s", command)
                    sent_success = True
                elif ev3.send(command):
                    logger.info("Sent: %s", command)
                    sent_success = True
                else:
                    logger.error("Failed to send: %s", command)
                
                if sent_success:
                    # Update history and physical state tracking
                    prev_base64_image = base64_image
                    last_command = command
                    if command == "open_gripper":
                        gripper_open = True
                    elif command == "close_gripper":
                        gripper_open = False
                else:
                    prev_base64_image = base64_image
                    last_command = "none"
            else:
                prev_base64_image = base64_image
                last_command = "none"

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
    parser.add_argument("--api-base", default=None, help="Custom API base URL (e.g. http://localhost:11434/v1 for Ollama).")
    parser.add_argument(
        "--api-type",
        choices=("openai", "gemini"),
        default=None,
        help="API protocol/format to use. If not specified, automatically inferred from model name ('gpt' -> openai, others -> gemini)."
    )
    parser.add_argument("--api-timeout", type=float, default=60.0, help="VLM API connection timeout in seconds.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
