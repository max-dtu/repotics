"""
Task / Subtask mission system.

A ``Task`` is an ordered sequence of ``Subtask`` objects.  Each subtask has its
own goal condition, step budget, timing, and optional policy.  The Agent works
through them in order via ``Agent.run_task()``.

Quick example
-------------
::

    from robot import Task, Subtask, Command

    task = (
        Task("pick and place")
        .then(Subtask(
            name="approach",
            description="Come within 1 cm of the ball",
            goal=lambda state: any(
                d["class_name"] == "ball" and d["w"] > 80
                for d in state["detections"]
            ),
            max_steps=60,
            step_delay=0.3,
        ))
        .then(Subtask(
            name="grab",
            description="Close the gripper around the ball",
            goal=lambda state: state["object_count"] == 0,   # ball hidden in gripper
            policy=lambda state: Command.CLOSE_GRIPPER,
            max_steps=10,
            step_delay=0.2,
        ))
        .then(Subtask(
            name="deliver",
            description="Move to the red square",
            goal=lambda state: "red_square" in state["objects_by_class"],
            max_steps=80,
        ))
    )

    result = robot.run_task(task)

    for r in result.subtask_results:
        status = "✓" if r.success else ("–" if r.skipped else "✗")
        print(f"  {status} {r.subtask.name}: {r.steps_taken} step(s)")

    print("Mission success:", result.success)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subtask
# ---------------------------------------------------------------------------

@dataclass
class Subtask:
    """
    A single goal unit within a ``Task``.

    Parameters
    ----------
    name:
        Short identifier shown in logs and results.
    goal:
        Termination condition — anything ``Evaluator.is_goal_reached`` accepts:

        * Callable ``(state: dict) -> bool`` — most flexible.
        * Any value handled by ``Evaluator._evaluate_goal``.

    description:
        Human-readable explanation shown in logs.  Optional.
    max_steps:
        Hard cap on the number of sense-act cycles before the subtask is
        declared failed.
    step_delay:
        Seconds to pause between steps so the robot and camera can settle.
    policy:
        Command-selection strategy.  ``"random"`` or a callable
        ``(state: dict) -> Command``.
    on_success:
        Optional callback ``(result: SubtaskResult) -> None`` called when the
        subtask goal is reached.
    on_failure:
        Optional callback ``(result: SubtaskResult) -> None`` called when
        max_steps is exhausted without reaching the goal.
    """

    name:        str
    goal:        Any
    description: str   = ""
    max_steps:   int   = 100
    step_delay:  float = 0.5
    policy:      Any   = "random"
    on_success:  Callable | None = field(default=None, repr=False)
    on_failure:  Callable | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"Subtask '{self.name}': max_steps must be >= 1.")
        if self.step_delay < 0:
            raise ValueError(f"Subtask '{self.name}': step_delay must be >= 0.")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SubtaskResult:
    """
    Outcome of a single subtask execution.

    Attributes
    ----------
    subtask:     The ``Subtask`` that was executed.
    success:     ``True`` if the goal was reached within ``max_steps``.
    steps_taken: Number of sense-act cycles performed (0 if skipped).
    skipped:     ``True`` if the task's ``fail_fast`` triggered before this
                 subtask was attempted.
    """

    subtask:     Subtask
    success:     bool
    steps_taken: int
    skipped:     bool = False

    def __str__(self) -> str:
        if self.skipped:
            return f"[–] {self.subtask.name} (skipped)"
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.subtask.name} ({self.steps_taken} step(s))"


@dataclass
class TaskResult:
    """
    Outcome of a full ``Task`` execution.

    Attributes
    ----------
    task:             The ``Task`` that was executed.
    subtask_results:  Ordered list of results, one per subtask.
    success:          ``True`` iff every subtask succeeded (none skipped, none failed).
    """

    task:            "Task"
    subtask_results: list[SubtaskResult]
    success:         bool

    def __str__(self) -> str:
        lines = [f"Task '{self.task.name}' — {'SUCCESS' if self.success else 'FAILED'}"]
        for r in self.subtask_results:
            lines.append(f"  {r}")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        print(self)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class Task:
    """
    An ordered sequence of ``Subtask`` objects forming a complete mission.

    Parameters
    ----------
    name:
        Human-readable mission name (shown in logs and results).
    subtasks:
        Optional initial list of subtasks.  Use ``.then()`` to add more.
    fail_fast:
        If ``True`` (default), stop executing after the first subtask that
        fails to reach its goal.  Remaining subtasks are marked *skipped*.
        If ``False``, attempt all subtasks regardless of earlier failures.

    Usage
    -----
    ::

        task = Task("demo") \\
            .then(Subtask("s1", goal=..., description="step 1")) \\
            .then(Subtask("s2", goal=..., description="step 2"))
    """

    def __init__(
        self,
        name: str,
        subtasks: list[Subtask] | None = None,
        fail_fast: bool = True,
    ) -> None:
        self.name      = name
        self.fail_fast = fail_fast
        self._subtasks: list[Subtask] = list(subtasks) if subtasks else []

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def then(self, subtask: Subtask) -> "Task":
        """
        Append *subtask* to the mission and return ``self`` for chaining.

        ::

            task = Task("mission") \\
                .then(Subtask("approach", goal=...)) \\
                .then(Subtask("grab",     goal=...))
        """
        if not isinstance(subtask, Subtask):
            raise TypeError(f"Expected Subtask, got {type(subtask).__name__!r}")
        self._subtasks.append(subtask)
        return self

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def subtasks(self) -> list[Subtask]:
        """Ordered list of subtasks (read-only copy)."""
        return list(self._subtasks)

    def __len__(self) -> int:
        return len(self._subtasks)

    def __repr__(self) -> str:
        return (
            f"Task(name={self.name!r}, subtasks={len(self._subtasks)}, "
            f"fail_fast={self.fail_fast})"
        )
