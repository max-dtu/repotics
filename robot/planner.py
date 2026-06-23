r"""
Planner — converts a (from, to) pair into a sequence of Commands.

``find_path`` is the primary entry point.  The stub returns an empty list;
replace ``_compute_path`` to add a real planning algorithm.

Planned approaches (pick one when ready)
-----------------------------------------
* **Grid / A\*** — discretise the camera frame into a grid, run A* from the
  pixel centroid nearest ``from_pos`` to the one nearest ``to_pos``,
  then translate grid steps → Commands (e.g. right→RIGHT, up→FORWARD).
* **RRT / sampling-based** — useful when the workspace has obstacles derived
  from detections.
* **Waypoint following** — if the path is known a priori, hardcode waypoints
  and select the closest command at each step.
* **Model-based** — pass the frame + goal to a vision-language model and
  parse a command sequence from its output.

Position types
--------------
``from_pos`` and ``to_pos`` are deliberately untyped.  They can be:

* ``(x, y)`` pixel coordinates
* ``(cx, cy)`` centroid values returned by ``Evaluator.assess()``
* A class-name string resolved at planning time via detections
* A dict ``{"class": "cup", "instance": 0}``
"""

import logging
import math
from typing import Any, Callable

from .commands import Command

logger = logging.getLogger(__name__)


class Planner:
    """
    Plans a path from *from_pos* to *to_pos* as a sequence of ``Command`` values.

    Parameters
    ----------
    evaluator:
        An ``Evaluator`` instance used to read the current world state during
        planning (e.g. obstacle positions).  May be ``None`` if the planner
        operates purely on given coordinates.
    spatial:
        A ``Spatial`` instance for geometric queries (free-space grid,
        approach points, obstacle centroids).  Optional.
    """

    def __init__(self, evaluator=None, spatial=None) -> None:
        self._evaluator = evaluator
        self._spatial   = spatial

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_path(self, from_pos: Any, to_pos: Any) -> list[Command]:
        """
        Returns an ordered list of ``Command`` values that, when executed
        sequentially, should move the robot from *from_pos* to *to_pos*.

        Returns an empty list while the stub backend is active.

        Parameters
        ----------
        from_pos:
            Current position (pixel coords, class name, dict, …).
        to_pos:
            Target position in the same format as *from_pos*.
        """
        logger.info(f"Planning path: {from_pos!r} → {to_pos!r}")
        path = self._compute_path(from_pos, to_pos)
        logger.info(f"Path found: {[c.value for c in path] if path else '(stub — empty)'}")
        return path

    def find_best_path(
        self,
        from_pos: Any,
        to_pos: Any,
        *,
        cost_fn: Callable[[list[Command]], float] | None = None,
    ) -> list[Command]:
        """
        Like ``find_path`` but selects the lowest-cost path according to
        *cost_fn* when multiple candidate paths are available.

        Parameters
        ----------
        from_pos, to_pos:
            Same as ``find_path``.
        cost_fn:
            ``(path: list[Command]) -> float`` — lower is better.  Defaults
            to path length (fewest steps).  Custom examples::

                # Penalise turns
                def fewest_turns(path):
                    turns = sum(1 for a, b in zip(path, path[1:]) if a != b)
                    return len(path) + 5 * turns

                # Penalise obstacle crossings (needs real planner)
                def avoid_obstacles(path):
                    return len(path) + 10 * count_crossings(path, obstacles)

                robot.find_best_path(src, dst, cost_fn=fewest_turns)

        Returns
        -------
        The lowest-cost path found, or ``[]`` while the stub is active.

        .. note::
           The stub ``_compute_path`` only generates one candidate, so
           ``find_best_path`` and ``find_path`` are equivalent until a real
           planner that produces multiple candidates is wired in.
        """
        logger.info(f"find_best_path: {from_pos!r} → {to_pos!r}")

        # With a real planner: generate N candidates, score each, return best.
        # Stub: single candidate.
        path = self._compute_path(from_pos, to_pos)

        if path:
            effective_cost = cost_fn(path) if cost_fn else float(len(path))
            logger.info(
                f"find_best_path: path length={len(path)} cost={effective_cost:.3f}"
            )
        else:
            logger.info("find_best_path: stub returned empty path.")

        return path

    def is_path_clear(
        self,
        from_pos: Any,
        to_pos: Any,
        state: dict | None = None,
    ) -> bool:
        """
        Check whether a straight line from *from_pos* to *to_pos* is free of
        detected obstacles.

        Returns ``True`` when no detection centroid lies within a corridor of
        half-width ``corridor_px`` around the line segment.

        .. note::
           Requires ``self._spatial`` to be set (wired automatically via
           ``Robot``).  Returns ``True`` (optimistic) if spatial is unavailable
           or no state can be resolved.

        ::

            if planner.is_path_clear(robot_pos, target_pos):
                robot.find_path(robot_pos, target_pos)
            else:
                robot.find_best_path(robot_pos, target_pos,
                                     cost_fn=lambda p: len(p) + 10*crossings(p))
        """
        if self._spatial is None:
            logger.debug("is_path_clear: no Spatial instance; returning True (optimistic).")
            return True

        if state is None and self._evaluator is not None:
            state = self._evaluator.assess()

        detections = (state or {}).get("detections", [])
        if not detections:
            return True

        fx, fy = from_pos if isinstance(from_pos, tuple) else (0, 0)
        tx, ty = to_pos   if isinstance(to_pos,   tuple) else (0, 0)
        corridor_px = 20   # half-width of the clearance corridor

        dx, dy = tx - fx, ty - fy
        seg_len = math.sqrt(dx ** 2 + dy ** 2)
        if seg_len < 1e-6:
            return True

        for det in detections:
            cx, cy = det["cx"], det["cy"]
            # Perpendicular distance from centroid to the line segment
            t = max(0.0, min(1.0, ((cx - fx) * dx + (cy - fy) * dy) / (seg_len ** 2)))
            proj_x = fx + t * dx
            proj_y = fy + t * dy
            perp = math.sqrt((cx - proj_x) ** 2 + (cy - proj_y) ** 2)
            if perp < corridor_px:
                logger.info(
                    f"is_path_clear: obstacle '{det.get('class_name','?')}' "
                    f"at ({cx},{cy}) blocks path (perp={perp:.1f}px)."
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Extension point
    # ------------------------------------------------------------------

    def _compute_path(self, from_pos: Any, to_pos: Any) -> list[Command]:
        """
        Override to implement a real path-planning algorithm.

        The method receives the raw *from_pos* / *to_pos* values passed by the
        caller.  Use ``self._evaluator.assess()`` to read the current world
        state (obstacle centroids, etc.) if needed.

        Must return an ordered ``list[Command]`` (may be empty).

        Example skeleton for pixel-coordinate A*::

            def _compute_path(self, from_pos, to_pos):
                state = self._evaluator.assess() if self._evaluator else {}
                obstacles = {(d["cx"], d["cy"]) for d in state.get("detections", [])}
                # ... run A* on a grid derived from camera resolution ...
                # ... translate grid path to Commands ...
                return commands
        """
        # --- STUB: replace with real path planning ---
        return []
