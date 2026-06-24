"""
Evaluator — interprets camera detections as robot state and checks goal conditions.

The Evaluator sits between the camera's Detector and the Agent's control loop.
It answers two questions on every iteration:

1. **What is the current state?**   ``assess() -> dict``
2. **Is the goal reached?**         ``is_goal_reached(goal) -> bool``

Goal representation
-------------------
The *goal* parameter is intentionally untyped so callers can use whatever
representation suits the task:

* A string label:  ``"red_block_in_zone"``
* A dict of target conditions:  ``{"class": "cup", "min_confidence": 0.8}``
* A callable predicate:  ``lambda state: state["object_count"] == 0``
* A target bounding-box region:  ``{"cx_range": (100, 200), "cy_range": (50, 150)}``

Implement ``_evaluate_goal`` (or subclass and override) to add real goal logic.
The stub always returns ``False`` so the Agent loop runs to ``max_steps``.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Reads the latest detections from a ``Detector`` instance and evaluates
    whether a given goal condition has been met.

    Parameters
    ----------
    detector:
        A ``camera.detector.Detector`` instance (shared with the Camera object).
    """

    def __init__(self, detector) -> None:
        self._detector = detector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self) -> dict:
        """
        Returns a snapshot of the current perceived world state.

        The dict always contains:

        ``detections``
            Raw list of detection dicts from the camera Detector.
        ``object_count``
            Number of detected objects.
        ``objects_by_class``
            ``{class_name: [det, …]}`` mapping for quick lookup.
        ``centroids``
            ``[(cx, cy), …]`` list for spatial reasoning.

        Add more derived fields here as the project grows (e.g. gripper state,
        occupancy grid, distance estimates).
        """
        # Exclude the "path" pseudo-entry injected by get_detections() for rendering.
        detections = [d for d in self._detector.get_detections() if d.get("class_name") != "path"]

        objects_by_class: dict[str, list] = {}
        for det in detections:
            cls = det.get("class_name", "unknown")
            objects_by_class.setdefault(cls, []).append(det)

        centroids = [(d.get("cx", 0), d.get("cy", 0)) for d in detections]

        state = {
            "detections":      detections,
            "object_count":    len(detections),
            "objects_by_class": objects_by_class,
            "centroids":       centroids,
        }

        logger.debug(f"State assessed: {len(detections)} object(s) detected.")
        return state

    def is_goal_reached(self, goal: Any) -> bool:
        """
        Check whether *goal* has been satisfied given the current state.

        If *goal* is a **callable**, it is called with the result of
        ``assess()`` and its return value is used directly — this is the
        most flexible option for custom logic::

            robot.run_until_complete(
                goal=lambda state: any(
                    d["class_name"] == "cup" for d in state["detections"]
                )
            )

        All other goal types are passed to ``_evaluate_goal`` which is a
        stub returning ``False``.  Override that method to add real evaluation.
        """
        state = self.assess()

        if callable(goal):
            result = bool(goal(state))
            if result:
                logger.info("Goal reached (callable predicate returned True).")
            return result

        result = self._evaluate_goal(goal, state)
        if result:
            logger.info(f"Goal reached: {goal!r}")
        return result

    # ------------------------------------------------------------------
    # Extension point
    # ------------------------------------------------------------------

    def _evaluate_goal(self, goal: Any, state: dict) -> bool:
        """
        Override to implement goal-specific evaluation logic.

        Parameters
        ----------
        goal:   The goal object passed to ``is_goal_reached``.
        state:  The dict produced by ``assess()`` for this iteration.

        Returns ``True`` when the goal is satisfied.

        Examples
        --------
        *Target class present*::

            def _evaluate_goal(self, goal, state):
                if isinstance(goal, str):
                    return goal in state["objects_by_class"]
                return False

        *Target centroid in bounding region*::

            def _evaluate_goal(self, goal, state):
                if isinstance(goal, dict):
                    x0, x1 = goal.get("cx_range", (0, 9999))
                    y0, y1 = goal.get("cy_range", (0, 9999))
                    return any(x0 <= cx <= x1 and y0 <= cy <= y1
                               for cx, cy in state["centroids"])
                return False
        """
        logger.warning(
            f"_evaluate_goal: unhandled goal type {type(goal).__name__!r} — "
            "stub returns False. Override _evaluate_goal or pass a callable."
        )
        return False
