"""
Agent — the robot control loop.

The Agent composes Commander, Evaluator, and Planner into three top-level
operations:

* ``step(policy)``            — send one command, return it
* ``run_until_complete(goal)``— loop until goal or max_steps exhausted
* ``run_task(task)``          — execute an ordered sequence of Subtasks

Policies
--------
The *policy* parameter controls how the next command is selected:

* ``"random"`` (default) — ``random.choice(AVAILABLE_COMMANDS)``
* Any callable           — called with the current state dict, must return a Command::

      def my_policy(state: dict) -> Command:
          if state["object_count"] > 0:
              return Command.OPEN_GRIPPER
          return Command.FORWARD

      agent.step(policy=my_policy)

This makes it straightforward to swap in learned policies, rule-based
planners, or LLM-driven controllers without touching the loop logic.
"""

import random
import time
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Union

from .commands import Command, AVAILABLE_COMMANDS
from .commander import Commander
from .evaluator import Evaluator
from .planner import Planner
from .task import Task, Subtask, SubtaskResult, TaskResult

logger = logging.getLogger(__name__)

Policy = Union[str, Callable[[dict], Command]]


# ── Configuration & result types for path following ──────────────────────────

@dataclass
class PathFollowConfig:
    """Tunable thresholds for the closed-loop path follower."""

    waypoint_threshold_px: float = 25.0
    """Distance (px) within which a waypoint is considered reached."""

    stuck_threshold_px: float = 5.0
    """If the robot moves less than this many pixels across
    ``stuck_patience`` consecutive cycles, it is declared stuck."""

    stuck_patience: int = 3
    """Number of consecutive low-movement cycles before stuck recovery."""

    max_stuck_recoveries: int = 3
    """Max recovery manoeuvres per waypoint before giving up on it."""

    max_detection_retries: int = 5
    """Max consecutive frames where the robot is not detected before
    the current waypoint is aborted."""

    detection_backoff_start: float = 0.3
    """Initial sleep (seconds) after a detection dropout."""

    alignment_angle_deg: float = 25.0
    """Angle error (degrees) above which the robot turns instead of
    driving forward."""

    max_attempts_per_waypoint: int = 60
    """Hard cap on iterations for a single waypoint."""

    command_send_retries: int = 2
    """Extra attempts to resend a command after the first failure."""

    max_consecutive_send_failures: int = 3
    """Abort the entire path after this many back-to-back send failures."""


@dataclass
class PathFollowResult:
    """Outcome of a closed-loop path execution."""

    commands_sent: int = 0
    waypoints_reached: int = 0
    total_waypoints: int = 0
    aborted: bool = False
    abort_reason: str = ""
    elapsed_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return not self.aborted and self.waypoints_reached == self.total_waypoints

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else ("ABORTED" if self.aborted else "INCOMPLETE")
        return (
            f"PathFollowResult({status}: "
            f"{self.waypoints_reached}/{self.total_waypoints} waypoints, "
            f"{self.commands_sent} cmds, {self.elapsed_seconds:.1f}s"
            f"{', reason=' + self.abort_reason if self.abort_reason else ''})"
        )


# ── Agent ────────────────────────────────────────────────────────────────────

class Agent:
    """
    Orchestrates Commander, Evaluator, and Planner into a control loop.

    Parameters
    ----------
    commander:  Sends commands to the robot hardware.
    evaluator:  Reads camera state, checks goal conditions.
    planner:    Plans sequences of commands between positions.
    """

    def __init__(
        self,
        commander: Commander,
        evaluator: Evaluator,
        planner: Planner,
    ) -> None:
        self._commander = commander
        self._evaluator = evaluator
        self._planner   = planner
        self.config     = PathFollowConfig()

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def step(self, policy: Policy = "random") -> Command:
        """
        Assess the current state, select one command via *policy*, send it,
        and return it.

        Parameters
        ----------
        policy:
            ``"random"`` or a callable ``(state: dict) -> Command``.
        """
        state   = self._evaluator.assess()
        command = self._select_command(policy, state)
        self._commander.send(command)
        return command

    # ------------------------------------------------------------------
    # Full control loop
    # ------------------------------------------------------------------

    def run_until_complete(
        self,
        goal: Any,
        *,
        policy: Policy = "random",
        max_steps: int = 100,
        step_delay: float = 0.5,
    ) -> bool:
        """
        Repeatedly assess → check goal → step until the goal is reached or
        *max_steps* is exhausted.

        Parameters
        ----------
        goal:
            Anything ``Evaluator.is_goal_reached`` understands — a string, dict,
            or a callable ``(state: dict) -> bool``.
        policy:
            Command-selection strategy (default: ``"random"``).
        max_steps:
            Hard cap on the number of steps to prevent infinite loops.
        step_delay:
            Seconds to pause between steps (gives the robot and camera time
            to react before the next assessment).

        Returns
        -------
        ``True`` if the goal was reached, ``False`` if max_steps was exhausted.
        """
        success, _ = self._run_until_complete_tracked(
            goal, policy=policy, max_steps=max_steps, step_delay=step_delay
        )
        return success

    # ------------------------------------------------------------------
    # Multi-subtask mission
    # ------------------------------------------------------------------

    def run_task(self, task: Task) -> TaskResult:
        """
        Execute each ``Subtask`` in *task* in order.

        For every subtask the agent:

        1. Logs the subtask name and description.
        2. Calls the internal tracked loop (assess → check goal → step).
        3. Records a ``SubtaskResult`` with success flag and step count.
        4. Calls ``on_success`` / ``on_failure`` callbacks if provided.
        5. If the subtask failed and ``task.fail_fast=True``, marks all
           remaining subtasks as *skipped* and stops early.

        Parameters
        ----------
        task:
            A ``Task`` instance built with ``Task.then(Subtask(...))``.  See
            ``robot.task`` for the full API.

        Returns
        -------
        A ``TaskResult`` with per-subtask outcomes and an overall success flag.
        """
        logger.info(
            f"\n{'='*60}\n"
            f"  TASK: {task.name}  ({len(task)} subtask(s), "
            f"fail_fast={task.fail_fast})"
            f"\n{'='*60}"
        )

        subtask_results: list[SubtaskResult] = []
        failed = False

        for i, subtask in enumerate(task.subtasks, 1):
            # If a previous subtask failed and fail_fast is on, skip the rest
            if failed and task.fail_fast:
                result = SubtaskResult(
                    subtask=subtask,
                    success=False,
                    steps_taken=0,
                    skipped=True,
                )
                subtask_results.append(result)
                logger.info(
                    f"  [{i}/{len(task)}] {subtask.name!r} — SKIPPED "
                    f"(fail_fast triggered)"
                )
                continue

            desc = f" — {subtask.description}" if subtask.description else ""
            logger.info(
                f"\n  [{i}/{len(task)}] Subtask: {subtask.name!r}{desc}\n"
                f"  goal={subtask.goal!r}  max_steps={subtask.max_steps}  "
                f"step_delay={subtask.step_delay}s"
            )

            success, steps = self._run_until_complete_tracked(
                subtask.goal,
                policy=subtask.policy,
                max_steps=subtask.max_steps,
                step_delay=subtask.step_delay,
                label=subtask.name,
            )

            result = SubtaskResult(
                subtask=subtask,
                success=success,
                steps_taken=steps,
            )
            subtask_results.append(result)

            if success:
                logger.info(
                    f"  [{i}/{len(task)}] {subtask.name!r} — SUCCESS "
                    f"({steps} step(s))"
                )
                if subtask.on_success:
                    try:
                        subtask.on_success(result)
                    except Exception as e:
                        logger.error(
                            f"on_success callback for {subtask.name!r} raised: {e}"
                        )
            else:
                failed = True
                logger.warning(
                    f"  [{i}/{len(task)}] {subtask.name!r} — FAILED "
                    f"(exhausted {steps} step(s))"
                )
                if subtask.on_failure:
                    try:
                        subtask.on_failure(result)
                    except Exception as e:
                        logger.error(
                            f"on_failure callback for {subtask.name!r} raised: {e}"
                        )

        overall = all(r.success for r in subtask_results)
        logger.info(
            f"\n{'='*60}\n"
            f"  TASK '{task.name}' — {'SUCCESS' if overall else 'FAILED'}\n"
            f"{'='*60}"
        )
        return TaskResult(task=task, subtask_results=subtask_results, success=overall)

    # ------------------------------------------------------------------
    # Path execution
    # ------------------------------------------------------------------

    def find_path(
        self,
        from_pos: Any,
        to_pos: Any,
        *,
        step_delay: float = 0.5,
        execute: bool = True,
        config: PathFollowConfig | None = None,
    ) -> list[Command]:
        """
        Plan a path from *from_pos* to *to_pos* and optionally execute it.

        Parameters
        ----------
        from_pos, to_pos:
            Start and end positions (pixel coords, class names, dicts, …).
        step_delay:
            Pause between executed commands (seconds).  Ignored if
            ``execute=False``.
        execute:
            If ``True`` (default), send each command in the plan sequentially.
            If ``False``, return the plan without executing it.
        config:
            Optional ``PathFollowConfig`` with tunable thresholds.
            Uses sensible defaults if not provided.

        Returns
        -------
        The list of planned commands (may be empty if the planner stub is active).
        """
        path = self._planner.find_path(from_pos, to_pos)

        if not path:
            logger.info("find_path: planner returned an empty path (stub active?).")
            return path

        if execute:
            cfg = config or self.config
            detector = getattr(self._evaluator, "_detector", None)
            waypoints = getattr(detector, "_current_path", None) if detector else None

            if waypoints and len(waypoints) > 1:
                result = self._execute_closed_loop(
                    waypoints, step_delay=step_delay, config=cfg
                )
                logger.info(f"Closed-loop result: {result}")
            else:
                logger.info(f"Executing {len(path)}-step path in OPEN-LOOP mode...")
                for i, command in enumerate(path, 1):
                    logger.info(f"  path step {i}/{len(path)}: {command.value!r}")
                    self._commander.send(command)
                    if step_delay > 0:
                        time.sleep(step_delay)
                logger.info("Path execution complete.")

        return path

    def find_best_path(
        self,
        from_pos: Any,
        to_pos: Any,
        *,
        cost_fn: Callable[[list[Command]], float] | None = None,
        step_delay: float = 0.5,
        execute: bool = True,
    ) -> list[Command]:
        """
        Plan a path using Planner.find_best_path, and optionally execute it.
        """
        path = self._planner.find_best_path(from_pos, to_pos, cost_fn=cost_fn)

        if not path:
            logger.info("find_best_path: planner returned an empty path (stub active?).")
            return path

        if execute:
            logger.info(f"Executing {len(path)}-step best path...")
            for i, command in enumerate(path, 1):
                logger.info(f"  path step {i}/{len(path)}: {command.value!r}")
                self._commander.send(command)
                if step_delay > 0:
                    time.sleep(step_delay)
            logger.info("Path execution complete.")

        return path

    # ------------------------------------------------------------------
    # Closed-loop path follower (robust)
    # ------------------------------------------------------------------

    def _execute_closed_loop(
        self,
        waypoints: list,
        *,
        step_delay: float = 0.5,
        config: PathFollowConfig,
    ) -> PathFollowResult:
        """
        Navigate through *waypoints* using real-time camera feedback.

        Features:
        - Stuck detection with recovery manoeuvres
        - Detection dropout handling with exponential backoff
        - Command send failure handling with retries
        - Waypoint skipping when the robot overshoots
        - Structured progress logging
        - Graceful abort with detailed result
        """
        cfg = config
        result = PathFollowResult(total_waypoints=len(waypoints) - 1)
        t0 = time.monotonic()

        consecutive_send_failures = 0
        nav_waypoints = list(enumerate(waypoints))[1:]  # skip start position

        logger.info(
            f"Closed-loop path follow: {result.total_waypoints} waypoint(s), "
            f"thresholds: wp={cfg.waypoint_threshold_px}px, "
            f"stuck={cfg.stuck_threshold_px}px/{cfg.stuck_patience} cycles, "
            f"align={cfg.alignment_angle_deg}°"
        )

        wp_list_idx = 0  # index into nav_waypoints

        while wp_list_idx < len(nav_waypoints):
            wp_idx_orig, wp = nav_waypoints[wp_list_idx]
            wx, wy = wp[0], wp[1]
            wp_num = wp_list_idx + 1
            wp_total = len(nav_waypoints)

            logger.info(
                f"── Waypoint {wp_num}/{wp_total} at ({wx}, {wy}) ──"
            )

            # Per-waypoint state
            position_history: list[tuple[float, float]] = []
            stuck_recoveries = 0
            detection_misses = 0
            attempt = 0

            while attempt < cfg.max_attempts_per_waypoint:
                attempt += 1

                # ── 1. Assess current state ──────────────────────────────
                state = self._evaluator.assess()
                objects_by_class = state.get("objects_by_class", {})

                car_keys = [k for k in objects_by_class.keys() if k.startswith("Car")]
                if not car_keys:
                    detection_misses += 1
                    if detection_misses > cfg.max_detection_retries:
                        logger.error(
                            f"WP {wp_num}: Robot lost for {detection_misses} "
                            f"consecutive frames. Skipping waypoint."
                        )
                        break
                    backoff = cfg.detection_backoff_start * (2 ** (detection_misses - 1))
                    backoff = min(backoff, 5.0)  # cap at 5s
                    logger.warning(
                        f"WP {wp_num}: Robot not detected "
                        f"({detection_misses}/{cfg.max_detection_retries}). "
                        f"Waiting {backoff:.1f}s…"
                    )
                    time.sleep(backoff)
                    continue

                # Robot detected — reset miss counter
                detection_misses = 0

                car_det = objects_by_class[car_keys[0]][0]
                cx = car_det.get("cx")
                cy = car_det.get("cy")
                heading = car_det.get("heading")

                if cx is None or cy is None:
                    logger.warning(f"WP {wp_num}: Invalid centroid. Retrying…")
                    time.sleep(0.2)
                    continue

                # ── 2. Check waypoint proximity ──────────────────────────
                dx = wx - cx
                dy = wy - cy
                dist = math.sqrt(dx**2 + dy**2)

                if dist < cfg.waypoint_threshold_px:
                    logger.info(
                        f"✓ Waypoint {wp_num}/{wp_total} reached! "
                        f"(dist={dist:.1f}px, {attempt} iterations)"
                    )
                    result.waypoints_reached += 1
                    break

                # ── 3. Waypoint skipping (overshoot detection) ───────────
                if wp_list_idx + 1 < len(nav_waypoints):
                    next_wp = nav_waypoints[wp_list_idx + 1][1]
                    dist_to_next = math.sqrt(
                        (next_wp[0] - cx)**2 + (next_wp[1] - cy)**2
                    )
                    if dist_to_next < dist * 0.6:
                        logger.info(
                            f"⏭ Skipping waypoint {wp_num} (overshot): "
                            f"dist_current={dist:.1f}px, dist_next={dist_to_next:.1f}px"
                        )
                        result.waypoints_reached += 1
                        break

                # ── 4. Stuck detection ───────────────────────────────────
                position_history.append((cx, cy))
                if len(position_history) > cfg.stuck_patience:
                    position_history.pop(0)

                if len(position_history) == cfg.stuck_patience:
                    total_movement = 0.0
                    for i in range(1, len(position_history)):
                        px, py = position_history[i - 1]
                        qx, qy = position_history[i]
                        total_movement += math.sqrt((qx - px)**2 + (qy - py)**2)

                    if total_movement < cfg.stuck_threshold_px:
                        stuck_recoveries += 1
                        if stuck_recoveries > cfg.max_stuck_recoveries:
                            logger.error(
                                f"WP {wp_num}: Stuck recovery exhausted "
                                f"({cfg.max_stuck_recoveries} attempts). "
                                f"Skipping waypoint."
                            )
                            break

                        logger.warning(
                            f"WP {wp_num}: Robot stuck (moved {total_movement:.1f}px "
                            f"in {cfg.stuck_patience} cycles). "
                            f"Recovery attempt {stuck_recoveries}/{cfg.max_stuck_recoveries}…"
                        )
                        # Recovery: turn randomly + nudge forward
                        recovery_turn = random.choice([Command.LEFT, Command.RIGHT])
                        for cmd in [recovery_turn, recovery_turn, Command.FORWARD]:
                            ok = self._send_with_retry(cmd, cfg)
                            result.commands_sent += 1
                            if not ok:
                                consecutive_send_failures += 1
                            else:
                                consecutive_send_failures = 0
                            time.sleep(step_delay)

                        if consecutive_send_failures >= cfg.max_consecutive_send_failures:
                            result.aborted = True
                            result.abort_reason = (
                                f"Too many consecutive send failures "
                                f"({consecutive_send_failures})"
                            )
                            result.elapsed_seconds = time.monotonic() - t0
                            logger.error(f"PATH ABORTED: {result.abort_reason}")
                            return result

                        position_history.clear()
                        continue

                # ── 5. Compute steering command ──────────────────────────
                if heading is None:
                    logger.warning(
                        f"WP {wp_num}: No heading vector. Driving FORWARD blind."
                    )
                    cmd = Command.FORWARD
                else:
                    hx, hy = heading[0], heading[1]
                    vx, vy = hx - cx, hy - cy
                    mag_v = math.sqrt(vx**2 + vy**2)

                    if mag_v < 1e-6:
                        cmd = Command.FORWARD
                    else:
                        vx, vy = vx / mag_v, vy / mag_v
                        mag_t = math.sqrt(dx**2 + dy**2)
                        tx, ty = dx / mag_t, dy / mag_t

                        cross = vx * ty - vy * tx
                        dot = vx * tx + vy * ty
                        angle_err = math.atan2(cross, dot)

                        if abs(angle_err) > math.radians(cfg.alignment_angle_deg):
                            cmd = Command.RIGHT if angle_err > 0 else Command.LEFT
                            logger.info(
                                f"WP {wp_num}: Aligning "
                                f"(err={math.degrees(angle_err):.1f}°) → {cmd.value}"
                            )
                        else:
                            cmd = Command.FORWARD
                            logger.info(
                                f"WP {wp_num}: Aligned, driving "
                                f"(dist={dist:.1f}px) → {cmd.value}"
                            )

                # ── 6. Send command with retry ───────────────────────────
                ok = self._send_with_retry(cmd, cfg)
                result.commands_sent += 1

                if ok:
                    consecutive_send_failures = 0
                else:
                    consecutive_send_failures += 1
                    if consecutive_send_failures >= cfg.max_consecutive_send_failures:
                        result.aborted = True
                        result.abort_reason = (
                            f"Too many consecutive send failures "
                            f"({consecutive_send_failures})"
                        )
                        result.elapsed_seconds = time.monotonic() - t0
                        logger.error(f"PATH ABORTED: {result.abort_reason}")
                        return result

                detector = getattr(self._evaluator, "_detector", None) if self._evaluator else None
                if detector and hasattr(detector, "wait_for_fresh_detections"):
                    # Sleep slightly to let the command transmit and begin/finish execution
                    time.sleep(0.3)
                    # Capture current frame count
                    cnt = detector.get_detections_counter()
                    # Wait for a fresh frame processed after this count
                    detector.wait_for_fresh_detections(cnt, timeout=2.0)
                else:
                    time.sleep(step_delay)

            else:
                # max_attempts_per_waypoint exhausted
                logger.warning(
                    f"WP {wp_num}: Exhausted {cfg.max_attempts_per_waypoint} "
                    f"attempts without reaching waypoint."
                )

            # Progress summary for this waypoint
            elapsed = time.monotonic() - t0
            logger.info(
                f"── WP {wp_num} done | "
                f"reached={result.waypoints_reached}/{result.total_waypoints} | "
                f"cmds={result.commands_sent} | "
                f"elapsed={elapsed:.1f}s ──"
            )

            wp_list_idx += 1

        result.elapsed_seconds = time.monotonic() - t0

        if result.success:
            logger.info(
                f"✓ Closed-loop path follow COMPLETE: "
                f"{result.waypoints_reached}/{result.total_waypoints} waypoints, "
                f"{result.commands_sent} commands, {result.elapsed_seconds:.1f}s"
            )
        else:
            logger.warning(f"Closed-loop path follow finished: {result}")

        return result

    def _send_with_retry(self, cmd: Command, cfg: PathFollowConfig) -> bool:
        """Send a command, retrying up to ``cfg.command_send_retries`` extra times."""
        for attempt in range(1 + cfg.command_send_retries):
            ok = self._commander.send(cmd)
            if ok:
                return True
            if attempt < cfg.command_send_retries:
                logger.warning(
                    f"Command send retry {attempt + 1}/{cfg.command_send_retries} "
                    f"for {cmd.value!r}…"
                )
                time.sleep(0.2)
        return False

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_until_complete_tracked(
        self,
        goal: Any,
        *,
        policy: Policy = "random",
        max_steps: int = 100,
        step_delay: float = 0.5,
        label: str = "",
    ) -> tuple[bool, int]:
        """
        Internal loop that returns ``(success, steps_taken)``.
        Used by both ``run_until_complete`` and ``run_task``.
        """
        prefix = f"[{label}] " if label else ""
        policy_label = policy if isinstance(policy, str) else "<fn>"
        logger.info(
            f"{prefix}Loop started | max_steps={max_steps} "
            f"policy={policy_label!r}"
        )

        for step_n in range(1, max_steps + 1):
            if self._evaluator.is_goal_reached(goal):
                logger.info(f"{prefix}Goal reached after {step_n - 1} step(s).")
                return True, step_n - 1

            state   = self._evaluator.assess()
            command = self._select_command(policy, state)
            self._commander.send(command)

            logger.info(
                f"{prefix}step {step_n:>3}/{max_steps} | "
                f"cmd={command.value!r} | objects={state['object_count']}"
            )

            if step_delay > 0:
                time.sleep(step_delay)

        logger.warning(
            f"{prefix}Exhausted max_steps={max_steps} without reaching goal."
        )
        return False, max_steps

    def _select_command(self, policy: Policy, state: dict) -> Command:
        if policy == "random":
            return random.choice(AVAILABLE_COMMANDS)
        if callable(policy):
            cmd = policy(state)
            if not isinstance(cmd, Command):
                raise TypeError(
                    f"Policy callable must return a Command, got {type(cmd).__name__!r}"
                )
            return cmd
        raise ValueError(
            f"Unknown policy {policy!r}. "
            f"Use 'random' or a callable (state: dict) -> Command."
        )
