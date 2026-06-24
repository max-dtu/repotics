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

import heapq
import logging
import math
from typing import Any, Callable

from .commands import Command

logger = logging.getLogger(__name__)

# ── Direction mapping ────────────────────────────────────────────────────────
# Maps (dc_sign, dr_sign) → Command, where:
#   dc = column delta: +1 = object moves right in image, -1 = left
#   dr = row    delta: +1 = object moves down  in image, -1 = up
#
# TUNING: if the robot turns/drives the wrong way, swap Command values here.
# Examples:
#   Camera faces the scene from above, robot faces "up" in image:
#     (+1, 0) → RIGHT, (-1, 0) → LEFT, (0, +1) → BACKWARD, (0, -1) → FORWARD  ← DEFAULT
#   Camera rotated 90° clockwise:
#     (+1, 0) → BACKWARD, (-1, 0) → FORWARD, (0, +1) → LEFT,  (0, -1) → RIGHT
DIRECTION_MAP = {
    (+1,  0): Command.RIGHT,
    (-1,  0): Command.LEFT,
    ( 0, +1): Command.BACKWARD,
    ( 0, -1): Command.FORWARD,
}
# ─────────────────────────────────────────────────────────────────────────────



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

    def _resolve_coordinate(
        self,
        pos: Any,
        state: dict,
        use_heading: bool = False,
    ) -> "tuple[int, int] | None":
        """Resolve *pos* to pixel (x, y).

        Parameters
        ----------
        use_heading:
            When True and *pos* is an ``object_N`` string, return the original
            click coordinate (heading/destination) recorded for that object
            rather than its current centroid.  Use this for the *to_pos* arg
            so the path ends at the user-intended target point, not the object
            centre.
        """
        if isinstance(pos, tuple) and len(pos) == 2:
            return int(pos[0]), int(pos[1])

        if isinstance(pos, dict):
            if "cx" in pos and "cy" in pos:
                return int(pos["cx"]), int(pos["cy"])
            cls = pos.get("class")
            if cls:
                dets = state.get("objects_by_class", {}).get(cls, [])
                idx = pos.get("instance", 0)
                if idx < len(dets):
                    return int(dets[idx]["cx"]), int(dets[idx]["cy"])

        if isinstance(pos, str):
            # Check for click heading if this is being used as an endpoint
            if use_heading and pos.startswith("object_"):
                # First check if the assessed state has the dynamic heading
                dets = state.get("objects_by_class", {}).get(pos, [])
                if dets and "heading" in dets[0]:
                    heading = dets[0]["heading"]
                    logger.info(
                        f"Resolved {pos!r} to_pos via dynamic heading: {heading}"
                    )
                    return int(heading[0]), int(heading[1])

                try:
                    obj_id = int(pos.split("_", 1)[1])
                    detector = getattr(self._evaluator, "_detector", None)
                    if detector and hasattr(detector, "get_click_heading"):
                        heading = detector.get_click_heading(obj_id)
                        if heading is not None:
                            logger.info(
                                f"Resolved {pos!r} to_pos via click heading fallback: {heading}"
                            )
                            return heading
                except (ValueError, IndexError):
                    pass

            dets = state.get("objects_by_class", {}).get(pos, [])
            if dets:
                return int(dets[0]["cx"]), int(dets[0]["cy"])

        return None

    def _compute_path(self, from_pos: Any, to_pos: Any) -> list[Command]:
        state = self._evaluator.assess() if self._evaluator else {}
        detections = state.get("detections", [])

        # 1. Resolve coordinates
        # from_pos → centroid of the object (where the robot starts)
        # to_pos   → click heading if available (where the user pointed), else centroid
        p1 = self._resolve_coordinate(from_pos, state, use_heading=False)
        p2 = self._resolve_coordinate(to_pos,   state, use_heading=True)

        if not p1 or not p2:

            logger.warning(f"Could not resolve path coordinates: from_pos={from_pos!r} -> {p1!r}, to_pos={to_pos!r} -> {p2!r}")
            return []

        x1, y1 = p1
        x2, y2 = p2

        # 2. Get frame dimensions dynamically
        w_orig, h_orig = 640, 480
        if self._evaluator and hasattr(self._evaluator, "_detector"):
            detector = self._evaluator._detector
            if hasattr(detector, "_reader") and hasattr(detector._reader, "get_dimensions"):
                w_orig, h_orig = detector._reader.get_dimensions()

        # 3. Setup grid dimensions (32 cols x 24 rows) and calculate dynamic cell sizes
        grid_w, grid_h = 32, 24
        cell_w = w_orig / grid_w
        cell_h = h_orig / grid_h

        grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

        # Convert start/end coordinates to grid indices
        col1 = min(grid_w - 1, max(0, int(x1 / cell_w)))
        row1 = min(grid_h - 1, max(0, int(y1 / cell_h)))
        col2 = min(grid_w - 1, max(0, int(x2 / cell_w)))
        row2 = min(grid_h - 1, max(0, int(y2 / cell_h)))

        # 4. Mark obstacle cells
        # Obstacles are any detections whose class_name is NOT from_pos and NOT to_pos
        for det in detections:
            cls_name = det.get("class_name")
            if cls_name == "path" or cls_name == from_pos or cls_name == to_pos:
                continue

            ox, oy, ow, oh = det.get("x", 0), det.get("y", 0), det.get("w", 0), det.get("h", 0)
            margin = 10
            bx1 = max(0, int((ox - margin) / cell_w))
            by1 = max(0, int((oy - margin) / cell_h))
            bx2 = min(grid_w - 1, int((ox + ow + margin) / cell_w))
            by2 = min(grid_h - 1, int((oy + oh + margin) / cell_h))

            for r in range(by1, by2 + 1):
                for c in range(bx1, bx2 + 1):
                    grid[r][c] = 1  # Blocked

        # Feasibility check: make sure start/end coordinates are always unblocked
        grid[row1][col1] = 0
        grid[row2][col2] = 0

        # 5. A* Search
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        start = (row1, col1)
        goal = (row2, col2)

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {start: None}
        cost_so_far = {start: 0.0}

        found = False
        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                found = True
                break

            r, c = current
            # 8-way connectivity
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    if grid[nr][nc] == 0:
                        step_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                        neighbors.append(((nr, nc), step_cost))

            for next_node, step_cost in neighbors:
                new_cost = cost_so_far[current] + step_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(next_node, goal)
                    heapq.heappush(open_set, (priority, next_node))
                    came_from[next_node] = current

        # 6. Translate path to waypoints and Commands
        waypoints = []
        commands = []
        if found:
            path_cells = []
            curr = goal
            while curr is not None:
                path_cells.append(curr)
                curr = came_from[curr]
            path_cells.reverse()

            for r, c in path_cells:
                wp_x = int((c + 0.5) * cell_w)
                wp_y = int((r + 0.5) * cell_h)
                waypoints.append([wp_x, wp_y])

            # Translate step-by-step grid transitions → Commands.
            # For diagonal A* steps only one command is emitted (dominant axis).
            # The goal cell itself generates no command — only the transition *into* it does.
            for i in range(len(path_cells) - 1):
                r, c = path_cells[i]
                nr, nc = path_cells[i + 1]

                dc = nc - c   # column delta (+right, -left)
                dr = nr - r   # row    delta (+down/backward, -up/forward)

                # Pick dominant axis (break ties with horizontal)
                if abs(dc) >= abs(dr):
                    cmd = DIRECTION_MAP.get((+1 if dc > 0 else -1, 0))
                else:
                    cmd = DIRECTION_MAP.get((0, +1 if dr > 0 else -1))

                if cmd is not None:
                    commands.append(cmd)
        else:
            logger.warning("No path found via grid A*.")
            # Set direct line fallback path
            waypoints = [[x1, y1], [x2, y2]]

        # 7. Save path on detector instance if wired
        if self._evaluator and hasattr(self._evaluator, "_detector"):
            detector = self._evaluator._detector
            if hasattr(detector, "set_current_path"):
                detector.set_current_path(waypoints)

        return commands
