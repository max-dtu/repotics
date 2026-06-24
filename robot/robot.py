"""
Robot — thin facade over Commander, Evaluator, Planner, and Agent.

Usage
-----
::

    from camera import Camera
    from robot  import Robot

    camera = Camera()
    robot  = Robot(detector=camera._detector)

    robot.step()                            # one random command
    robot.run_until_complete("goal")        # loop until done
    robot.find_path((0, 0), (100, 80))     # plan + execute path

Wiring a real backend
---------------------
::

    def serial_backend(command):
        port.write((command.value + "\\n").encode())

    robot = Robot(detector=camera._detector, backend=serial_backend)
"""

import logging
from typing import Any, Callable, Union

from .commands import Command, AVAILABLE_COMMANDS
from .commander import Commander
from .evaluator import Evaluator
from .planner import Planner
from .agent import Agent, Policy
from .task import Task, TaskResult
from .spatial import Spatial

logger = logging.getLogger(__name__)


class Robot:
    """
    Facade that composes Commander, Evaluator, Planner, and Agent.

    All public methods delegate to the appropriate subsystem so the REPL
    (and future application code) has a single, flat API surface.

    Parameters
    ----------
    detector:
        ``camera.detector.Detector`` instance **shared** with the Camera object.
        Detections must be flowing (``camera.start_detection()`` called) before
        ``assess()`` or ``run_until_complete()`` will see real results.
    backend:
        Optional callable ``(command: Command) -> None`` for the Commander.
        Defaults to a log-only stub.
    """

    def __init__(self, detector, backend=None) -> None:
        self._commander = Commander(backend=backend)
        self._evaluator = Evaluator(detector)
        self._spatial   = Spatial(detector)
        self._planner   = Planner(evaluator=self._evaluator, spatial=self._spatial)
        self._agent     = Agent(self._commander, self._evaluator, self._planner)

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def send(self, command: Union[Command, str]) -> None:
        """
        Send a single command to the robot.

        Accepts either a ``Command`` enum value or its string equivalent::

            robot.send(Command.FORWARD)
            robot.send("forward")          # convenience
        """
        if isinstance(command, str):
            command = Command(command)
        self._commander.send(command)

    # ------------------------------------------------------------------
    # Agent loop
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
        return self._agent.step(policy=policy)

    def run_until_complete(
        self,
        goal: Any,
        *,
        policy: Policy = "random",
        max_steps: int = 100,
        step_delay: float = 0.5,
    ) -> bool:
        """
        Loop: assess → check goal → step, until goal reached or *max_steps* hit.

        Parameters
        ----------
        goal:
            Termination condition.  Options:

            * Callable ``(state: dict) -> bool`` — most flexible::

                  robot.run_until_complete(
                      goal=lambda s: "cup" in s["objects_by_class"]
                  )

            * Any value handled by ``Evaluator._evaluate_goal`` (override that
              method to add string/dict/region-based goals).

        policy:
            Command-selection strategy.  Default: ``"random"``.
        max_steps:
            Hard cap to prevent infinite loops.
        step_delay:
            Pause between steps in seconds (gives the robot and camera time to
            settle before re-assessing).

        Returns
        -------
        ``True`` if goal reached, ``False`` if max_steps exhausted.
        """
        return self._agent.run_until_complete(
            goal,
            policy=policy,
            max_steps=max_steps,
            step_delay=step_delay,
        )

    def run_task(self, task: Task) -> TaskResult:
        """
        Execute a multi-subtask mission and return the full result.

        Parameters
        ----------
        task:
            A ``Task`` built with ``Task.then(Subtask(...))``.  Example::

                from robot import Task, Subtask, Command

                task = (
                    Task("pick and place")
                    .then(Subtask(
                        name="approach",
                        description="Come within 1 cm of the ball",
                        goal=lambda s: any(
                            d["class_name"] == "ball" and d["w"] > 80
                            for d in s["detections"]
                        ),
                        max_steps=60,
                    ))
                    .then(Subtask(
                        name="grab",
                        description="Close gripper around ball",
                        goal=lambda s: s["object_count"] == 0,
                        policy=lambda s: Command.CLOSE_GRIPPER,
                        max_steps=10,
                    ))
                    .then(Subtask(
                        name="deliver",
                        description="Move to the red square",
                        goal=lambda s: "red_square" in s["objects_by_class"],
                        max_steps=80,
                    ))
                )

                result = robot.run_task(task)
                result.print_summary()

        Returns
        -------
        A ``TaskResult`` with per-subtask outcomes and an overall success flag.
        Call ``result.print_summary()`` for a formatted overview.
        """
        return self._agent.run_task(task)

    # ------------------------------------------------------------------
    # Path planning
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
        Plan (and optionally execute) a path from *from_pos* to *to_pos*.

        Returns the list of planned commands.  Currently delegates to the
        stub ``Planner._compute_path`` (returns ``[]``); replace that method
        to activate real path planning.

        Parameters
        ----------
        from_pos, to_pos:
            Start / end positions.  Accepted formats (all are stubs for now):

            * ``(x, y)`` pixel / centroid coordinates
            * A string class name: ``"cup"``
            * A detection dict returned by ``robot.assess()``

        execute:
            ``True`` (default) — send each planned command sequentially.
            ``False`` — return the plan without executing.
        """
        return self._agent.find_path(
            from_pos,
            to_pos,
            step_delay=step_delay,
            execute=execute,
        )

    def go_to_nearest_ball(
        self,
        *,
        ball_classes: list[str] | tuple[str, ...] = ("WhiteBall", "OrangeBall", "orange_ball", "white_ball"),
        robot_class: str = "Car",
        step_delay: float = 0.5,
        execute: bool = True,
    ) -> list[Command]:
        """
        Find the closest ball to the robot and drive to it.

        Searches across all *ball_classes*, picks the nearest one by centroid
        distance, plans an A* path, and optionally executes it.

        Parameters
        ----------
        ball_classes:
            Class names to consider as balls.
        robot_class:
            Class name used by the model for the robot (default: ``"Car"``).
        step_delay:
            Pause between commands in seconds.
        execute:
            ``True`` (default) to send commands; ``False`` to plan only.

        Returns
        -------
        List of planned commands, or ``[]`` if robot or balls are not detected.
        """
        state = self._evaluator.assess()
        robot_dets = state["objects_by_class"].get(robot_class, [])
        if not robot_dets:
            logger.warning(
                f"go_to_nearest_ball: robot not detected (looked for class '{robot_class}'). "
                f"Available classes: {list(state['objects_by_class'].keys())}"
            )
            return []
        robot_pos = (robot_dets[0]["cx"], robot_dets[0]["cy"])

        target = self._spatial.find_nearest_of_any(list(ball_classes), from_pos=robot_pos, state=state)
        if target is None:
            logger.warning("go_to_nearest_ball: no balls detected.")
            return []

        logger.info(
            f"go_to_nearest_ball: targeting '{target['class_name']}' "
            f"at ({target['cx']}, {target['cy']})"
        )
        return self.find_path(
            robot_pos,
            (target["cx"], target["cy"]),
            step_delay=step_delay,
            execute=execute,
        )

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
        Plan a path from *from_pos* to *to_pos* selecting the lowest-cost option according to *cost_fn*.
        Optionally execute it.
        """
        return self._agent.find_best_path(
            from_pos,
            to_pos,
            cost_fn=cost_fn,
            step_delay=step_delay,
            execute=execute,
        )

    def is_path_clear(
        self,
        from_pos: Any,
        to_pos: Any,
        state: dict | None = None,
    ) -> bool:
        """
        Check whether a straight line from *from_pos* to *to_pos* is free of detected obstacles.
        """
        return self._planner.is_path_clear(from_pos, to_pos, state=state)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def assess(self) -> dict:
        """
        Returns the current perceived world state from the camera::

            {
                "detections":       [...],   # raw detection dicts
                "object_count":     int,
                "objects_by_class": {class: [det, ...]},
                "centroids":        [(cx, cy), ...],
            }

        Requires ``camera.start_detection()`` to be called first.
        """
        return self._evaluator.assess()

    def is_goal_reached(self, goal: Any) -> bool:
        """Check whether *goal* is satisfied given the current camera state."""
        return self._evaluator.is_goal_reached(goal)

    @property
    def spatial(self) -> Spatial:
        """Geometric queries helper."""
        return self._spatial

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def available_commands(self) -> list[Command]:
        """All commands the robot understands."""
        return AVAILABLE_COMMANDS
