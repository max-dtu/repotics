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
from typing import Any, Callable, Union

from .commands import Command, AVAILABLE_COMMANDS
from .commander import Commander
from .evaluator import Evaluator
from .planner import Planner
from .task import Task, Subtask, SubtaskResult, TaskResult

logger = logging.getLogger(__name__)

Policy = Union[str, Callable[[dict], Command]]


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

        Returns
        -------
        The list of planned commands (may be empty if the planner stub is active).
        """
        path = self._planner.find_path(from_pos, to_pos)

        if not path:
            logger.info("find_path: planner returned an empty path (stub active?).")
            return path

        if execute:
            logger.info(f"Executing {len(path)}-step path...")
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
