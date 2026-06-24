"""
Spatial — frame-level geometric queries on detection state.

All methods are pure functions operating on detection dicts from
``robot.assess()`` or a direct list.  No threads, no hardware.

Detection dict format (from camera.Detector):
    {x, y, w, h, cx, cy, class_name, confidence}

Quick reference
---------------
::

    # Via robot facade (uses live detector):
    ball = robot.spatial.find_nearest("ball", from_pos=(320, 240))
    stop = robot.spatial.find_approach_point(ball, from_pos=(320, 240), gap_px=20)
    gap  = robot.spatial.gap_px(ball, gripper_det)
    robot.find_path((320, 240), stop)

    # On a one-shot captured frame:
    frame = camera.capture()
    # ... run detection on frame externally → detections list ...
    state = {"detections": detections, "objects_by_class": ..., ...}
    nearest = robot.spatial.find_nearest("cup", from_pos=(0, 0), state=state)
"""

import math
import logging
from typing import Any

logger = logging.getLogger(__name__)


class Spatial:
    """
    Geometric queries on detection state.

    Parameters
    ----------
    detector:
        Optional ``camera.detector.Detector`` instance.  When provided,
        methods that accept an optional ``state=`` argument will use the
        live detector output when no explicit state is passed.
    """

    def __init__(self, detector=None) -> None:
        self._detector = detector

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _resolve_state(self, state: dict | None) -> dict:
        """Return *state* if given, otherwise fetch from the live detector."""
        if state is not None:
            return state
        if self._detector is not None:
            detections = self._detector.get_detections()
        else:
            detections = []
        objects_by_class: dict = {}
        for det in detections:
            cls = det.get("class_name", "unknown")
            objects_by_class.setdefault(cls, []).append(det)
        return {
            "detections":       detections,
            "object_count":     len(detections),
            "objects_by_class": objects_by_class,
            "centroids":        [(d.get("cx", 0), d.get("cy", 0)) for d in detections],
        }

    # ------------------------------------------------------------------
    # Primitive geometry
    # ------------------------------------------------------------------

    def distance_px(self, pos_a: tuple, pos_b: tuple) -> float:
        """Euclidean pixel distance between two ``(x, y)`` points."""
        return math.sqrt((pos_b[0] - pos_a[0]) ** 2 + (pos_b[1] - pos_a[1]) ** 2)

    def gap_px(self, det_a: dict, det_b: dict) -> float:
        """
        Shortest pixel distance between the **edges** of two bounding boxes.

        Returns ``0`` if the boxes overlap or touch.  Useful for "approach
        within N pixels" goals — the gap shrinks as the robot closes in.

        ::

            gap = spatial.gap_px(robot_det, ball_det)
            Subtask("approach", goal=lambda s: spatial.gap_px(...) < 20)
        """
        # Horizontal separation between edges
        ax0, ax1 = det_a["x"], det_a["x"] + det_a["w"]
        bx0, bx1 = det_b["x"], det_b["x"] + det_b["w"]
        h_gap = max(0.0, max(bx0 - ax1, ax0 - bx1))

        # Vertical separation between edges
        ay0, ay1 = det_a["y"], det_a["y"] + det_a["h"]
        by0, by1 = det_b["y"], det_b["y"] + det_b["h"]
        v_gap = max(0.0, max(by0 - ay1, ay0 - by1))

        if h_gap == 0 and v_gap == 0:
            return 0.0          # overlapping or touching
        if h_gap == 0:
            return v_gap
        if v_gap == 0:
            return h_gap
        return math.sqrt(h_gap ** 2 + v_gap ** 2)

    # ------------------------------------------------------------------
    # Detection queries
    # ------------------------------------------------------------------

    def find_all(
        self,
        class_name: str,
        state: dict | None = None,
    ) -> list[dict]:
        """
        Return all detections of *class_name*, or ``[]`` if none present.

        ::

            balls = spatial.find_all("ball")
        """
        s = self._resolve_state(state)
        return list(s.get("objects_by_class", {}).get(class_name, []))

    def rank_by_distance(
        self,
        from_pos: tuple,
        detections: list[dict],
    ) -> list[dict]:
        """
        Return *detections* sorted by ascending distance from *from_pos*.

        ::

            nearest_first = spatial.rank_by_distance((320, 240), all_balls)
        """
        return sorted(
            detections,
            key=lambda d: self.distance_px(from_pos, (d["cx"], d["cy"])),
        )

    def find_nearest(
        self,
        class_name: str,
        from_pos: tuple = (0, 0),
        state: dict | None = None,
    ) -> dict | None:
        """
        Return the detection of *class_name* closest to *from_pos*, or
        ``None`` if no detection of that class is present.

        ::

            ball = spatial.find_nearest("ball", from_pos=(320, 240))
            if ball:
                stop = spatial.find_approach_point(ball, from_pos=(320, 240), gap_px=20)
        """
        candidates = self.find_all(class_name, state)
        if not candidates:
            logger.debug(f"find_nearest: no detections of class '{class_name}'.")
            return None
        return self.rank_by_distance(from_pos, candidates)[0]

    def find_nearest_of_any(
        self,
        class_names: list[str],
        from_pos: tuple = (0, 0),
        state: dict | None = None,
    ) -> dict | None:
        """
        Return the detection closest to *from_pos* across all *class_names*, or
        ``None`` if none of those classes are present.

        ::

            ball = spatial.find_nearest_of_any(["orange_ball", "white_ball"], from_pos=(320, 240))
        """
        s = self._resolve_state(state)
        candidates = []
        for cls in class_names:
            candidates.extend(s.get("objects_by_class", {}).get(cls, []))
        if not candidates:
            logger.debug(f"find_nearest_of_any: no detections for classes {class_names}.")
            return None
        return self.rank_by_distance(from_pos, candidates)[0]

    def find_closest_pair(
        self,
        class_a: str,
        class_b: str,
        state: dict | None = None,
    ) -> tuple[dict | None, dict | None]:
        """
        Return the ``(A, B)`` detection pair with the smallest edge-to-edge gap.
        Returns ``(None, None)`` if either class has no detections.

        Useful for checking which ball is closest to which target zone::

            ball, zone = spatial.find_closest_pair("ball", "red_square")
        """
        all_a = self.find_all(class_a, state)
        all_b = self.find_all(class_b, state)
        if not all_a or not all_b:
            return None, None

        best_gap = math.inf
        best_pair: tuple[dict | None, dict | None] = (None, None)
        for a in all_a:
            for b in all_b:
                g = self.gap_px(a, b)
                if g < best_gap:
                    best_gap = g
                    best_pair = (a, b)

        logger.debug(
            f"find_closest_pair('{class_a}', '{class_b}'): "
            f"best gap = {best_gap:.1f} px"
        )
        return best_pair

    def is_in_region(self, detection: dict, region: dict) -> bool:
        """
        Return ``True`` if the detection's centroid lies inside *region*.

        *region* is a dict with keys ``x0``, ``y0``, ``x1``, ``y1``
        (pixel coordinates, top-left / bottom-right corners)::

            goal=lambda s: spatial.is_in_region(
                spatial.find_nearest("ball", state=s),
                {"x0": 100, "y0": 50, "x1": 220, "y1": 180},
            )
        """
        if detection is None:
            return False
        cx, cy = detection["cx"], detection["cy"]
        return (
            region["x0"] <= cx <= region["x1"]
            and region["y0"] <= cy <= region["y1"]
        )

    # ------------------------------------------------------------------
    # Approach geometry
    # ------------------------------------------------------------------

    def find_approach_point(
        self,
        target_det: dict,
        from_pos: tuple,
        gap_px: float = 20,
    ) -> tuple[int, int]:
        """
        Return ``(x, y)`` — the point that is *gap_px* pixels from the nearest
        edge of *target_det*, along the straight line from *from_pos* toward
        the target's centre.

        Use as the ``to_pos`` in ``find_path`` to implement "approach within
        N pixels" without colliding with the object::

            ball  = spatial.find_nearest("ball", from_pos=robot_pos)
            stop  = spatial.find_approach_point(ball, robot_pos, gap_px=20)
            robot.find_path(robot_pos, stop)

        Parameters
        ----------
        target_det:
            Detection dict of the object to approach.
        from_pos:
            ``(x, y)`` starting position (e.g. robot pixel location or frame
            centre ``(frame_w//2, frame_h//2)``).
        gap_px:
            Desired stop distance from the target's bounding-box edge.
        """
        cx, cy = target_det["cx"], target_det["cy"]
        fx, fy = from_pos

        dx, dy = cx - fx, cy - fy
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < 1e-6:
            logger.warning("find_approach_point: from_pos coincides with target centre.")
            return from_pos

        ux, uy = dx / dist, dy / dist

        # Approximate the distance from target centre to its edge along the
        # approach direction using a weighted box half-size.
        edge_dist = (target_det["w"] * abs(ux) + target_det["h"] * abs(uy)) / 2
        stop_dist = dist - edge_dist - gap_px

        if stop_dist <= 0:
            logger.warning(
                f"find_approach_point: already within gap_px={gap_px:.0f} px of "
                f"target '{target_det.get('class_name', '?')}'. Returning from_pos."
            )
            return from_pos

        return (int(fx + ux * stop_dist), int(fy + uy * stop_dist))

    # ------------------------------------------------------------------
    # Scale calibration
    # ------------------------------------------------------------------

    def calibrate_scale(self, det: dict, real_size_cm: float) -> float:
        """
        Compute a ``px_per_cm`` scale factor from a detection of a known object.

        Uses the bounding-box **width** as the reference dimension.  Store the
        returned value and pass it to ``estimate_distance_cm``::

            # A4 paper is 21 cm wide
            px_per_cm = spatial.calibrate_scale(paper_det, real_size_cm=21.0)
            gap_cm    = spatial.estimate_distance_cm(ball_det, zone_det, px_per_cm)

        Raises ``ValueError`` if the detection width is zero.
        """
        if det["w"] <= 0:
            raise ValueError("Detection width must be > 0 for scale calibration.")
        scale = det["w"] / real_size_cm
        logger.info(f"Scale calibrated: {scale:.2f} px/cm  (ref width {det['w']} px = {real_size_cm} cm)")
        return scale

    def estimate_distance_cm(
        self,
        det_a: dict,
        det_b: dict,
        px_per_cm: float,
    ) -> float:
        """
        Edge-to-edge gap in **centimetres** between two detections.

        Requires a ``px_per_cm`` value from ``calibrate_scale``::

            gap_cm = spatial.estimate_distance_cm(robot_det, ball_det, px_per_cm)
            Subtask("approach", goal=lambda s: ... < 1.0)   # 1 cm gap
        """
        return self.gap_px(det_a, det_b) / px_per_cm

    # ------------------------------------------------------------------
    # Free-space grid
    # ------------------------------------------------------------------

    def find_free_space(
        self,
        frame_w: int,
        frame_h: int,
        state: dict | None = None,
        cell_size: int = 32,
    ) -> list[tuple[int, int]]:
        """
        Return a list of ``(cx, cy)`` cell centres that are not covered by any
        detection bounding box.  Useful as candidate waypoints for path planning.

        Parameters
        ----------
        frame_w, frame_h:
            Camera frame dimensions in pixels (from ``camera.get_dimensions()``).
        cell_size:
            Grid resolution in pixels (default 32 px).
        state:
            Explicit state dict, or ``None`` to use the live detector.

        ::

            w, h  = camera.get_dimensions()
            cells = spatial.find_free_space(w, h)
            # Pass cells to a path planner as navigable waypoints
        """
        s = self._resolve_state(state)
        detections = s.get("detections", [])

        free: list[tuple[int, int]] = []
        for row in range(frame_h // cell_size):
            for col in range(frame_w // cell_size):
                ccx = col * cell_size + cell_size // 2
                ccy = row * cell_size + cell_size // 2
                occupied = any(
                    det["x"] <= ccx <= det["x"] + det["w"]
                    and det["y"] <= ccy <= det["y"] + det["h"]
                    for det in detections
                )
                if not occupied:
                    free.append((ccx, ccy))

        logger.debug(
            f"find_free_space: {len(free)}/{(frame_w // cell_size) * (frame_h // cell_size)} "
            f"cells free  (cell_size={cell_size}px)"
        )
        return free
