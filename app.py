from __future__ import annotations

import base64
import logging
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_UPLOAD_BYTES", str(250 * 1024 * 1024))
)

DEFAULT_BALL_DIAMETER_CM = 7.25
DEFAULT_HUE_TOLERANCE_PERCENT = 10.0
DEFAULT_SATURATION_TOLERANCE_PERCENT = 40.0
DEFAULT_VALUE_TOLERANCE_PERCENT = 40.0
CALIBRATION_DISTANCE_M = 1.0

DERIVATIVE_SPAN_FRAMES = max(
    1,
    int(os.environ.get("DERIVATIVE_SPAN_FRAMES", "3")),
)
MIN_METRIC_RADIUS_PX = float(
    os.environ.get("MIN_METRIC_RADIUS_PX", "2.5")
)
BOUNCE_MIN_CHANGE_RATE_MPS2 = float(
    os.environ.get("BOUNCE_MIN_CHANGE_RATE_MPS2", "30")
)
BOUNCE_MIN_ANGLE_DEG = float(
    os.environ.get("BOUNCE_MIN_ANGLE_DEG", "8")
)
BOUNCE_ROBUST_SIGMA_MULTIPLIER = float(
    os.environ.get("BOUNCE_ROBUST_SIGMA_MULTIPLIER", "5")
)
BOUNCE_MERGE_WINDOW_S = float(
    os.environ.get("BOUNCE_MERGE_WINDOW_S", "0.12")
)
BOUNCE_SUMMARY_EXCLUSION_S = float(
    os.environ.get("BOUNCE_SUMMARY_EXCLUSION_S", "0.15")
)

EPSILON = 1e-12


logging.basicConfig(
    level=getattr(
        logging,
        os.environ.get("LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    ),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)
logger = logging.getLogger("speedbowled")


class AppError(ValueError):
    """User-facing application error."""

    def __init__(
        self,
        message: str,
        status_code: int = 422,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


@dataclass
class BallDetection:
    """One tracked image-plane observation."""

    frame_index: int
    time_s: float
    center_x: float
    center_y: float
    radius_px: float


@dataclass
class BallCandidate:
    """One color/shape/motion-qualified ball candidate."""

    center: tuple[float, float]
    radius: float
    area: float
    circularity: float
    aspect_ratio: float
    motion_ratio: float
    descriptors: Optional[np.ndarray]


@dataclass
class DerivativeSeries:
    """Finite-difference values and provenance."""

    values: np.ndarray
    measured_mask: np.ndarray
    sources: list[str]
    span: int


def circular_hue_distance(
    hue: np.ndarray,
    target: int,
) -> np.ndarray:
    """Return OpenCV HSV hue distance with wrap-around."""
    raw = np.abs(
        hue.astype(np.int16)
        - int(target)
    )
    return np.minimum(
        raw,
        180 - raw,
    )


def build_hsv_mask(
    frame: np.ndarray,
    target_hsv: tuple[int, int, int],
    hue_tolerance_percent: float,
    saturation_tolerance_percent: float,
    value_tolerance_percent: float,
) -> np.ndarray:
    """Build an HSV mask from user-configurable percentage tolerances."""
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    hue, saturation, value = cv2.split(
        hsv
    )

    (
        target_hue,
        target_saturation,
        target_value,
    ) = target_hsv

    hue_tolerance = max(
        0,
        int(
            round(
                179.0
                * hue_tolerance_percent
                / 100.0
            )
        ),
    )

    saturation_tolerance = max(
        0,
        int(
            round(
                255.0
                * saturation_tolerance_percent
                / 100.0
            )
        ),
    )

    value_tolerance = max(
        0,
        int(
            round(
                255.0
                * value_tolerance_percent
                / 100.0
            )
        ),
    )

    saturation_lower = max(
        0,
        target_saturation
        - saturation_tolerance,
    )

    saturation_upper = min(
        255,
        target_saturation
        + saturation_tolerance,
    )

    value_lower = max(
        0,
        target_value
        - value_tolerance,
    )

    value_upper = min(
        255,
        target_value
        + value_tolerance,
    )

    selected = (
        (
            circular_hue_distance(
                hue,
                target_hue,
            )
            <= hue_tolerance
        )
        & (
            saturation
            >= saturation_lower
        )
        & (
            saturation
            <= saturation_upper
        )
        & (
            value
            >= value_lower
        )
        & (
            value
            <= value_upper
        )
    )

    mask = np.where(
        selected,
        255,
        0,
    ).astype(
        np.uint8
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )


def contour_motion_ratio(
    contour: np.ndarray,
    motion_mask: np.ndarray,
) -> float:
    """Return moving-pixel fraction inside a contour."""
    x, y, width, height = cv2.boundingRect(
        contour
    )

    if width <= 0 or height <= 0:
        return 0.0

    local = contour.copy()
    local[:, 0, 0] -= x
    local[:, 0, 1] -= y

    contour_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.drawContours(
        contour_mask,
        [local],
        -1,
        255,
        -1,
    )

    motion_roi = motion_mask[
        y : y + height,
        x : x + width,
    ]

    overlap = cv2.bitwise_and(
        motion_roi,
        contour_mask,
    )

    return (
        cv2.countNonZero(
            overlap
        )
        / max(
            cv2.countNonZero(
                contour_mask
            ),
            1,
        )
    )


def descriptors_near_circle(
    center: tuple[float, float],
    radius: float,
    keypoints,
    descriptors: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Extract SIFT descriptors close to a candidate."""
    if descriptors is None or not keypoints:
        return None

    center_vector = np.asarray(
        center,
        dtype=np.float32,
    )

    search_radius = max(
        radius * 2.0,
        6.0,
    )

    indices = [
        index
        for index, keypoint in enumerate(
            keypoints
        )
        if np.linalg.norm(
            np.asarray(
                keypoint.pt,
                dtype=np.float32,
            )
            - center_vector
        )
        <= search_radius
    ]

    if not indices:
        return None

    return np.asarray(
        descriptors[indices],
        dtype=np.float32,
    )


class BallTracker:
    """Track a moving, circular blob of the calibrated ball color."""

    def __init__(
        self,
        target_hsv: tuple[int, int, int],
        hue_tolerance_percent: float,
        saturation_tolerance_percent: float,
        value_tolerance_percent: float,
    ) -> None:
        if not hasattr(
            cv2,
            "SIFT_create",
        ):
            raise AppError(
                "This OpenCV build does not include SIFT.",
                500,
            )

        self.target_hsv = target_hsv
        self.hue_tolerance_percent = hue_tolerance_percent
        self.saturation_tolerance_percent = saturation_tolerance_percent
        self.value_tolerance_percent = value_tolerance_percent

        self.sift = cv2.SIFT_create(
            nfeatures=900,
            contrastThreshold=0.02,
            edgeThreshold=10,
        )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_L2,
            crossCheck=False,
        )

        self.previous_gray: Optional[np.ndarray] = None
        self.previous_keypoints = []
        self.previous_descriptors: Optional[np.ndarray] = None

        self.tracked_center: Optional[np.ndarray] = None
        self.tracked_radius: Optional[float] = None
        self.tracked_descriptors: Optional[np.ndarray] = None

        self.tracked_velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.missed_frames = 0
        self.stats: Counter[str] = Counter()

    def _alignment(
        self,
        keypoints,
        descriptors: Optional[np.ndarray],
    ) -> tuple[
        Optional[np.ndarray],
        list,
    ]:
        """Estimate global image motion with SIFT."""
        if (
            self.previous_descriptors is None
            or descriptors is None
            or len(
                self.previous_descriptors
            )
            < 2
            or len(
                descriptors
            )
            < 2
        ):
            return None, []

        raw_matches = self.matcher.knnMatch(
            self.previous_descriptors,
            descriptors,
            k=2,
        )

        good = [
            pair[0]
            for pair in raw_matches
            if (
                len(pair) == 2
                and pair[0].distance
                < 0.75
                * pair[1].distance
            )
        ]

        if len(good) < 6:
            return None, good

        previous_points = np.float32(
            [
                self.previous_keypoints[
                    match.queryIdx
                ].pt
                for match in good
            ]
        )

        current_points = np.float32(
            [
                keypoints[
                    match.trainIdx
                ].pt
                for match in good
            ]
        )

        matrix, _ = cv2.estimateAffinePartial2D(
            previous_points,
            current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        return matrix, good

    def _motion_mask(
        self,
        current_gray: np.ndarray,
        alignment: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build a motion mask after optional SIFT compensation."""
        if alignment is None:
            previous = self.previous_gray
        else:
            height, width = (
                current_gray.shape
            )

            previous = cv2.warpAffine(
                self.previous_gray,
                alignment,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        difference = cv2.GaussianBlur(
            cv2.absdiff(
                current_gray,
                previous,
            ),
            (3, 3),
            0,
        )

        _, motion = cv2.threshold(
            difference,
            9,
            255,
            cv2.THRESH_BINARY,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        return cv2.dilate(
            motion,
            kernel,
            iterations=1,
        )

    def _candidates(
        self,
        mask: np.ndarray,
        motion_mask: Optional[np.ndarray],
        keypoints,
        descriptors: Optional[np.ndarray],
    ) -> list[BallCandidate]:
        """Find circular, color-qualified moving blobs."""
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[
            BallCandidate
        ] = []

        for contour in contours:
            area = float(
                cv2.contourArea(
                    contour
                )
            )

            perimeter = float(
                cv2.arcLength(
                    contour,
                    True,
                )
            )

            if (
                area < 2.0
                or perimeter <= EPSILON
            ):
                continue

            (
                center_x,
                center_y,
            ), radius = cv2.minEnclosingCircle(
                contour
            )

            radius = float(
                radius
            )

            if not (
                0.8
                <= radius
                <= 260.0
            ):
                continue

            _, _, width, height = (
                cv2.boundingRect(
                    contour
                )
            )

            aspect_ratio = (
                min(
                    width,
                    height,
                )
                / max(
                    width,
                    height,
                    1,
                )
            )

            if aspect_ratio < 0.42:
                continue

            circularity = (
                4.0
                * math.pi
                * area
                / (
                    perimeter
                    * perimeter
                )
            )

            if (
                radius >= 4.0
                and circularity < 0.45
            ):
                continue

            motion_ratio = 1.0

            if motion_mask is not None:
                motion_ratio = (
                    contour_motion_ratio(
                        contour,
                        motion_mask,
                    )
                )

                if motion_ratio < 0.01:
                    continue

            candidates.append(
                BallCandidate(
                    center=(
                        float(center_x),
                        float(center_y),
                    ),
                    radius=radius,
                    area=area,
                    circularity=circularity,
                    aspect_ratio=aspect_ratio,
                    motion_ratio=motion_ratio,
                    descriptors=(
                        descriptors_near_circle(
                            (
                                center_x,
                                center_y,
                            ),
                            radius,
                            keypoints,
                            descriptors,
                        )
                    ),
                )
            )

        return candidates

    def _descriptor_matches(
        self,
        previous: Optional[np.ndarray],
        current: Optional[np.ndarray],
    ) -> int:
        """Count good local SIFT descriptor matches."""
        if (
            previous is None
            or current is None
            or len(previous) < 2
            or len(current) < 2
        ):
            return 0

        raw = self.matcher.knnMatch(
            previous,
            current,
            k=2,
        )

        return sum(
            1
            for pair in raw
            if (
                len(pair) == 2
                and pair[0].distance
                < 0.75
                * pair[1].distance
            )
        )

    def _select(
        self,
        candidates: list[
            BallCandidate
        ],
    ) -> Optional[
        BallCandidate
    ]:
        """Associate one candidate with the active track."""
        if not candidates:
            return None

        if (
            self.tracked_center is None
            or self.tracked_radius is None
        ):
            return max(
                candidates,
                key=lambda candidate: (
                    candidate.aspect_ratio
                    + min(
                        candidate.motion_ratio,
                        1.0,
                    )
                    + max(
                        candidate.circularity,
                        0.0,
                    )
                    + min(
                        candidate.area
                        / 300.0,
                        1.0,
                    )
                ),
            )

        prediction = (
            self.tracked_center
            + self.tracked_velocity
            * (
                self.missed_frames
                + 1
            )
        )

        best = None
        best_score = -math.inf

        for candidate in candidates:
            center = np.asarray(
                candidate.center,
                dtype=np.float32,
            )

            distance = float(
                np.linalg.norm(
                    center
                    - prediction
                )
            )

            tracked_speed = float(
                np.linalg.norm(
                    self.tracked_velocity
                )
            )

            max_jump = max(
                30.0,
                candidate.radius
                * 10.0,
                tracked_speed
                * 4.0,
            )

            if distance > max_jump:
                continue

            radius_ratio = (
                candidate.radius
                / max(
                    self.tracked_radius,
                    EPSILON,
                )
            )

            if not (
                0.35
                <= radius_ratio
                <= 2.8
            ):
                continue

            descriptor_score = min(
                self._descriptor_matches(
                    self.tracked_descriptors,
                    candidate.descriptors,
                )
                / 5.0,
                1.0,
            )

            proximity = max(
                0.0,
                1.0
                - distance
                / max_jump,
            )

            radius_score = max(
                0.0,
                1.0
                - abs(
                    math.log(
                        max(
                            radius_ratio,
                            EPSILON,
                        )
                    )
                )
                / math.log(2.8),
            )

            score = (
                4.0
                * proximity
                + 2.0
                * radius_score
                + 1.5
                * descriptor_score
                + candidate.aspect_ratio
                + min(
                    candidate.motion_ratio,
                    1.0,
                )
            )

            if score > best_score:
                best_score = score
                best = candidate

        return best

    def update(
        self,
        frame: np.ndarray,
    ) -> Optional[
        tuple[
            float,
            float,
            float,
        ]
    ]:
        """Track the ball in one frame."""
        self.stats[
            "frames_processed"
        ] += 1

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        keypoints, descriptors = (
            self.sift.detectAndCompute(
                gray,
                None,
            )
        )

        color_mask = build_hsv_mask(
            frame,
            self.target_hsv,
            self.hue_tolerance_percent,
            self.saturation_tolerance_percent,
            self.value_tolerance_percent,
        )

        if self.previous_gray is None:
            candidates = self._candidates(
                color_mask,
                None,
                keypoints,
                descriptors,
            )
        else:
            alignment, _ = self._alignment(
                keypoints,
                descriptors,
            )

            if alignment is None:
                self.stats[
                    "camera_alignment_unavailable"
                ] += 1
            else:
                self.stats[
                    "camera_alignment_success"
                ] += 1

            candidates = self._candidates(
                color_mask,
                self._motion_mask(
                    gray,
                    alignment,
                ),
                keypoints,
                descriptors,
            )

        selected = self._select(
            candidates
        )

        self.previous_gray = gray
        self.previous_keypoints = (
            keypoints
        )
        self.previous_descriptors = (
            descriptors
        )

        if selected is None:
            self.missed_frames += 1

            self.stats[
                "frames_without_detection"
            ] += 1

            if self.missed_frames > 8:
                self.tracked_center = None
                self.tracked_radius = None
                self.tracked_descriptors = None
                self.tracked_velocity[:] = 0.0
                self.missed_frames = 0

                self.stats[
                    "track_resets"
                ] += 1

            return None

        new_center = np.asarray(
            selected.center,
            dtype=np.float32,
        )

        if (
            self.tracked_center
            is not None
        ):
            observed_velocity = (
                new_center
                - self.tracked_center
            ) / max(
                self.missed_frames
                + 1,
                1,
            )

            self.tracked_velocity = (
                0.35
                * self.tracked_velocity
                + 0.65
                * observed_velocity
            )

        self.tracked_center = (
            new_center
        )

        self.tracked_radius = (
            selected.radius
        )

        if (
            selected.descriptors
            is not None
            and len(
                selected.descriptors
            )
            >= 2
        ):
            self.tracked_descriptors = (
                selected.descriptors
            )

        self.missed_frames = 0

        self.stats[
            "frames_with_detection"
        ] += 1

        return (
            selected.center[0],
            selected.center[1],
            selected.radius,
        )

    def debug_summary(
        self,
    ) -> dict[str, Any]:
        """Return tracker diagnostics."""
        processed = max(
            self.stats[
                "frames_processed"
            ],
            1,
        )

        return {
            "frames_processed": (
                self.stats[
                    "frames_processed"
                ]
            ),
            "frames_with_detection": (
                self.stats[
                    "frames_with_detection"
                ]
            ),
            "frames_without_detection": (
                self.stats[
                    "frames_without_detection"
                ]
            ),
            "detection_rate": (
                self.stats[
                    "frames_with_detection"
                ]
                / processed
            ),
            "camera_alignment_success": (
                self.stats[
                    "camera_alignment_success"
                ]
            ),
            "camera_alignment_unavailable": (
                self.stats[
                    "camera_alignment_unavailable"
                ]
            ),
            "track_resets": (
                self.stats[
                    "track_resets"
                ]
            ),
        }


def parse_float_field(
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Parse one numeric form field."""
    raw = request.form.get(
        name
    )

    if raw is None:
        raise AppError(
            f"Missing field: {name}.",
            400,
        )

    try:
        value = float(
            raw
        )
    except ValueError as exc:
        raise AppError(
            f"{name} must be numeric.",
            400,
        ) from exc

    if (
        not math.isfinite(
            value
        )
        or not (
            minimum
            <= value
            <= maximum
        )
    ):
        raise AppError(
            f"{name} must be between "
            f"{minimum} and {maximum}.",
            400,
        )

    return value


def sample_hsv_at_point(
    image: np.ndarray,
    x: int,
    y: int,
) -> tuple[
    int,
    int,
    int,
]:
    """Sample ball HSV from a small click neighborhood."""
    height, width = (
        image.shape[
            :2
        ]
    )

    if not (
        0 <= x < width
        and 0 <= y < height
    ):
        raise AppError(
            "Selected point is outside the image.",
            400,
        )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    radius = 4

    patch = hsv[
        max(
            0,
            y - radius,
        ) : min(
            height,
            y + radius + 1,
        ),
        max(
            0,
            x - radius,
        ) : min(
            width,
            x + radius + 1,
        ),
    ].reshape(
        -1,
        3,
    ).astype(
        np.float64
    )

    angles = (
        patch[
            :,
            0,
        ]
        * (
            2.0
            * math.pi
            / 180.0
        )
    )

    angle = math.atan2(
        float(
            np.mean(
                np.sin(
                    angles
                )
            )
        ),
        float(
            np.mean(
                np.cos(
                    angles
                )
            )
        ),
    )

    if angle < 0:
        angle += (
            2.0
            * math.pi
        )

    hue = (
        int(
            round(
                angle
                * 180.0
                / (
                    2.0
                    * math.pi
                )
            )
        )
        % 180
    )

    saturation = int(
        round(
            float(
                np.median(
                    patch[
                        :,
                        1,
                    ]
                )
            )
        )
    )

    value = int(
        round(
            float(
                np.median(
                    patch[
                        :,
                        2,
                    ]
                )
            )
        )
    )

    return (
        hue,
        saturation,
        value,
    )


def calibrate_image(
    image: np.ndarray,
    x: int,
    y: int,
    ball_diameter_m: float,
    hue_tolerance_percent: float,
    saturation_tolerance_percent: float,
    value_tolerance_percent: float,
) -> dict[str, Any]:
    """Calibrate focal length from the selected ball at one metre."""
    target_hsv = sample_hsv_at_point(
        image,
        x,
        y,
    )

    mask = build_hsv_mask(
        image,
        target_hsv,
        hue_tolerance_percent,
        saturation_tolerance_percent,
        value_tolerance_percent,
    )

    (
        component_count,
        labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    selected_label = int(
        labels[
            y,
            x,
        ]
    )

    if selected_label == 0:
        click = np.asarray(
            [x, y],
            dtype=np.float64,
        )

        usable = [
            label
            for label in range(
                1,
                component_count,
            )
            if int(
                stats[
                    label,
                    cv2.CC_STAT_AREA,
                ]
            )
            >= 6
        ]

        if not usable:
            raise AppError(
                "No usable ball region was found. "
                "Click nearer the ball centre."
            )

        selected_label = min(
            usable,
            key=lambda label: (
                np.linalg.norm(
                    centroids[
                        label
                    ]
                    - click
                )
            ),
        )

    component = np.where(
        labels
        == selected_label,
        255,
        0,
    ).astype(
        np.uint8
    )

    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise AppError(
            "Could not measure the selected ball."
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    area = float(
        cv2.contourArea(
            contour
        )
    )

    perimeter = float(
        cv2.arcLength(
            contour,
            True,
        )
    )

    (
        center_x,
        center_y,
    ), enclosing_radius = (
        cv2.minEnclosingCircle(
            contour
        )
    )

    equivalent_radius = math.sqrt(
        max(
            area,
            1.0,
        )
        / math.pi
    )

    radius_px = (
        0.65
        * enclosing_radius
        + 0.35
        * equivalent_radius
        + 0.35
    )

    diameter_px = (
        2.0
        * radius_px
    )

    if diameter_px <= 2.0:
        raise AppError(
            "The ball is too small in the calibration image."
        )

    focal_length_px = (
        diameter_px
        * CALIBRATION_DISTANCE_M
        / ball_diameter_m
    )

    circularity = (
        4.0
        * math.pi
        * area
        / max(
            perimeter
            * perimeter,
            EPSILON,
        )
    )

    preview = image.copy()

    cv2.circle(
        preview,
        (
            round(
                center_x
            ),
            round(
                center_y
            ),
        ),
        round(
            radius_px
        ),
        (
            50,
            255,
            120,
        ),
        3,
        cv2.LINE_AA,
    )

    cv2.drawMarker(
        preview,
        (x, y),
        (
            255,
            255,
            255,
        ),
        cv2.MARKER_CROSS,
        22,
        2,
        cv2.LINE_AA,
    )

    preview_uri = None

    ok, encoded = cv2.imencode(
        ".jpg",
        preview,
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            84,
        ],
    )

    if ok:
        preview_uri = (
            "data:image/jpeg;base64,"
            + base64.b64encode(
                encoded.tobytes()
            ).decode(
                "ascii"
            )
        )

    bgr = cv2.cvtColor(
        np.uint8(
            [[target_hsv]]
        ),
        cv2.COLOR_HSV2BGR,
    )[0, 0]

    warnings = []

    if circularity < 0.60:
        warnings.append(
            "The detected region is only weakly circular. "
            "A cleaner image may improve calibration."
        )

    return {
        "focal_length_px": float(
            focal_length_px
        ),
        "ball_diameter_px": float(
            diameter_px
        ),
        "circularity": float(
            circularity
        ),
        "color": {
            "hsv": [
                int(
                    value
                )
                for value in target_hsv
            ],
            "rgb": [
                int(
                    bgr[2]
                ),
                int(
                    bgr[1]
                ),
                int(
                    bgr[0]
                ),
            ],
        },
        "preview_data_uri": (
            preview_uri
        ),
        "tolerances_percent": {
            "h": float(
                hue_tolerance_percent
            ),
            "s": float(
                saturation_tolerance_percent
            ),
            "v": float(
                value_tolerance_percent
            ),
        },
        "warnings": warnings,
    }


def finite_difference_at(
    times: np.ndarray,
    values: np.ndarray,
    index: int,
    span: int,
) -> tuple[
    Optional[np.ndarray],
    str,
]:
    """Calculate one derivative using finite differences only."""
    count = len(
        times
    )

    maximum_span = min(
        span,
        max(
            count - 1,
            0,
        ),
    )

    for current_span in range(
        maximum_span,
        0,
        -1,
    ):
        left = (
            index
            - current_span
        )
        right = (
            index
            + current_span
        )

        if (
            left >= 0
            and right < count
        ):
            delta_t = (
                times[
                    right
                ]
                - times[
                    left
                ]
            )

            if delta_t > EPSILON:
                return (
                    (
                        values[
                            right
                        ]
                        - values[
                            left
                        ]
                    )
                    / delta_t,
                    (
                        f"central_span_"
                        f"{current_span}"
                    ),
                )

    for current_span in range(
        maximum_span,
        0,
        -1,
    ):
        right = (
            index
            + current_span
        )

        if right < count:
            delta_t = (
                times[
                    right
                ]
                - times[
                    index
                ]
            )

            if delta_t > EPSILON:
                return (
                    (
                        values[
                            right
                        ]
                        - values[
                            index
                        ]
                    )
                    / delta_t,
                    (
                        f"forward_span_"
                        f"{current_span}"
                    ),
                )

    for current_span in range(
        maximum_span,
        0,
        -1,
    ):
        left = (
            index
            - current_span
        )

        if left >= 0:
            delta_t = (
                times[
                    index
                ]
                - times[
                    left
                ]
            )

            if delta_t > EPSILON:
                return (
                    (
                        values[
                            index
                        ]
                        - values[
                            left
                        ]
                    )
                    / delta_t,
                    (
                        f"backward_span_"
                        f"{current_span}"
                    ),
                )

    return (
        None,
        "unavailable",
    )


def differentiate(
    times: np.ndarray,
    values: np.ndarray,
    *,
    span: int,
) -> DerivativeSeries:
    """Differentiate with finite differences and retain provenance."""
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim == 1:
        values = values[
            :,
            None,
        ]

    derivatives = np.full(
        values.shape,
        np.nan,
        dtype=np.float64,
    )

    measured = np.zeros(
        len(
            times
        ),
        dtype=bool,
    )

    sources = [
        "unavailable"
    ] * len(
        times
    )

    for index in range(
        len(
            times
        )
    ):
        derivative, source = (
            finite_difference_at(
                times,
                values,
                index,
                span,
            )
        )

        if (
            derivative is not None
            and np.all(
                np.isfinite(
                    derivative
                )
            )
        ):
            derivatives[
                index
            ] = derivative

            measured[
                index
            ] = True

            sources[
                index
            ] = source

    valid = np.flatnonzero(
        measured
    )

    if len(valid):
        for index in np.flatnonzero(
            ~measured
        ):
            nearest = int(
                valid[
                    np.argmin(
                        np.abs(
                            valid
                            - index
                        )
                    )
                ]
            )

            derivatives[
                index
            ] = derivatives[
                nearest
            ]

            sources[
                index
            ] = (
                "fallback_nearest_"
                + sources[
                    nearest
                ]
            )

    elif len(times):
        derivatives[:] = 0.0

        sources = [
            "fallback_zero"
        ] * len(
            times
        )

    return DerivativeSeries(
        values=derivatives,
        measured_mask=measured,
        sources=sources,
        span=span,
    )


def proper_derivative_mask(
    series: DerivativeSeries,
) -> np.ndarray:
    """Accept only the full central finite-difference stencil."""
    expected = (
        f"central_span_"
        f"{series.span}"
    )

    finite = np.all(
        np.isfinite(
            series.values
        ),
        axis=1,
    )

    source_ok = np.asarray(
        [
            source == expected
            for source in series.sources
        ],
        dtype=bool,
    )

    return (
        series.measured_mask
        & finite
        & source_ok
    )


def metric_positions(
    detections: list[
        BallDetection
    ],
    focal_length_px: float,
    ball_radius_m: float,
    principal_x: float,
    principal_y: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Reconstruct 3-D ball positions with the calibrated pinhole camera."""
    times = np.asarray(
        [
            item.time_s
            for item in detections
        ],
        dtype=np.float64,
    )

    radii = np.asarray(
        [
            max(
                item.radius_px,
                0.25,
            )
            for item in detections
        ],
        dtype=np.float64,
    )

    x = np.asarray(
        [
            item.center_x
            for item in detections
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        [
            item.center_y
            for item in detections
        ],
        dtype=np.float64,
    )

    scale = (
        ball_radius_m
        / radii
    )

    positions = np.column_stack(
        (
            (
                x
                - principal_x
            )
            * scale,
            (
                y
                - principal_y
            )
            * scale,
            focal_length_px
            * scale,
        )
    )

    return (
        times,
        positions,
        radii,
    )


def radius_quality_mask(
    radii: np.ndarray,
    *,
    span: int,
) -> np.ndarray:
    """Require usable radii at central derivative endpoints."""
    result = np.zeros(
        len(
            radii
        ),
        dtype=bool,
    )

    for index in range(
        len(
            radii
        )
    ):
        left = (
            index
            - span
        )
        right = (
            index
            + span
        )

        if (
            left < 0
            or right
            >= len(
                radii
            )
        ):
            continue

        selected = radii[
            [
                left,
                index,
                right,
            ]
        ]

        result[
            index
        ] = bool(
            np.all(
                np.isfinite(
                    selected
                )
            )
            and np.all(
                selected
                >= MIN_METRIC_RADIUS_PX
            )
        )

    return result


def swing_rate(
    times: np.ndarray,
    velocity: DerivativeSeries,
    valid_velocity: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Calculate 3-D velocity direction-change rate in degrees/second."""
    values = np.full(
        len(
            times
        ),
        np.nan,
        dtype=np.float64,
    )

    valid = np.zeros(
        len(
            times
        ),
        dtype=bool,
    )

    span = (
        velocity.span
    )

    for index in range(
        len(
            times
        )
    ):
        left = (
            index
            - span
        )
        right = (
            index
            + span
        )

        if (
            left < 0
            or right
            >= len(
                times
            )
        ):
            continue

        if (
            not valid_velocity[
                left
            ]
            or not valid_velocity[
                right
            ]
        ):
            continue

        first = (
            velocity.values[
                left
            ]
        )

        second = (
            velocity.values[
                right
            ]
        )

        first_norm = float(
            np.linalg.norm(
                first
            )
        )

        second_norm = float(
            np.linalg.norm(
                second
            )
        )

        if (
            first_norm <= 0.25
            or second_norm <= 0.25
        ):
            continue

        cosine = float(
            np.clip(
                np.dot(
                    first,
                    second,
                )
                / (
                    first_norm
                    * second_norm
                ),
                -1.0,
                1.0,
            )
        )

        delta_t = (
            times[
                right
            ]
            - times[
                left
            ]
        )

        if delta_t <= EPSILON:
            continue

        values[
            index
        ] = (
            math.degrees(
                math.acos(
                    cosine
                )
            )
            / delta_t
        )

        valid[
            index
        ] = math.isfinite(
            values[
                index
            ]
        )

    return (
        values,
        valid,
    )


def detect_bounce_events(
    times: np.ndarray,
    velocity: DerivativeSeries,
    valid_velocity: np.ndarray,
    detections: list[
        BallDetection
    ],
) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],
    dict[str, Any],
]:
    """Detect abrupt velocity-vector changes without trajectory fitting."""
    valid_indices = np.flatnonzero(
        valid_velocity
    )

    if len(
        valid_indices
    ) < 3:
        return (
            [],
            {
                "candidate_count": 0,
                "threshold_mps2": (
                    BOUNCE_MIN_CHANGE_RATE_MPS2
                ),
                "median_change_rate_mps2": 0.0,
                "robust_sigma_mps2": 0.0,
            },
        )

    candidates: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for (
        previous_index,
        current_index,
    ) in zip(
        valid_indices[
            :-1
        ],
        valid_indices[
            1:
        ],
    ):
        delta_t = (
            times[
                current_index
            ]
            - times[
                previous_index
            ]
        )

        if (
            delta_t <= EPSILON
            or delta_t > 0.25
        ):
            continue

        previous_velocity = (
            velocity.values[
                previous_index
            ]
        )

        current_velocity = (
            velocity.values[
                current_index
            ]
        )

        previous_speed = float(
            np.linalg.norm(
                previous_velocity
            )
        )

        current_speed = float(
            np.linalg.norm(
                current_velocity
            )
        )

        if (
            previous_speed <= 0.25
            or current_speed <= 0.25
        ):
            continue

        delta_velocity = (
            current_velocity
            - previous_velocity
        )

        change_mps = float(
            np.linalg.norm(
                delta_velocity
            )
        )

        change_rate_mps2 = (
            change_mps
            / delta_t
        )

        cosine = float(
            np.clip(
                np.dot(
                    previous_velocity,
                    current_velocity,
                )
                / (
                    previous_speed
                    * current_speed
                ),
                -1.0,
                1.0,
            )
        )

        angle_deg = math.degrees(
            math.acos(
                cosine
            )
        )

        candidates.append(
            {
                "index": int(
                    current_index
                ),
                "time_s": float(
                    times[
                        current_index
                    ]
                ),
                "frame_index": int(
                    detections[
                        current_index
                    ].frame_index
                ),
                "change_mps": (
                    change_mps
                ),
                "change_rate_mps2": (
                    change_rate_mps2
                ),
                "angle_deg": (
                    angle_deg
                ),
                "speed_before_kmph": (
                    previous_speed
                    * 3.6
                ),
                "speed_after_kmph": (
                    current_speed
                    * 3.6
                ),
            }
        )

    if not candidates:
        return (
            [],
            {
                "candidate_count": 0,
                "threshold_mps2": (
                    BOUNCE_MIN_CHANGE_RATE_MPS2
                ),
                "median_change_rate_mps2": 0.0,
                "robust_sigma_mps2": 0.0,
            },
        )

    rates = np.asarray(
        [
            candidate[
                "change_rate_mps2"
            ]
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    median_rate = float(
        np.median(
            rates
        )
    )

    mad = float(
        np.median(
            np.abs(
                rates
                - median_rate
            )
        )
    )

    robust_sigma = (
        1.4826
        * mad
    )

    threshold = max(
        BOUNCE_MIN_CHANGE_RATE_MPS2,
        median_rate
        + BOUNCE_ROBUST_SIGMA_MULTIPLIER
        * robust_sigma,
    )

    abrupt = [
        candidate
        for candidate in candidates
        if (
            candidate[
                "change_rate_mps2"
            ]
            >= threshold
            and (
                candidate[
                    "angle_deg"
                ]
                >= BOUNCE_MIN_ANGLE_DEG
                or candidate[
                    "change_rate_mps2"
                ]
                >= 1.5
                * threshold
            )
        )
    ]

    merged: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for candidate in abrupt:
        if not merged:
            merged.append(
                candidate
            )
            continue

        previous = merged[
            -1
        ]

        if (
            candidate[
                "time_s"
            ]
            - previous[
                "time_s"
            ]
            <= BOUNCE_MERGE_WINDOW_S
        ):
            if (
                candidate[
                    "change_rate_mps2"
                ]
                > previous[
                    "change_rate_mps2"
                ]
            ):
                merged[
                    -1
                ] = candidate
        else:
            merged.append(
                candidate
            )

    events = []

    for number, candidate in enumerate(
        merged,
        start=1,
    ):
        event = dict(
            candidate
        )

        event.update(
            {
                "number": number,
                "label": (
                    f"Bounce {number}"
                ),
                "type": (
                    "bounce_or_abrupt_velocity_change"
                ),
            }
        )

        events.append(
            event
        )

    return (
        events,
        {
            "candidate_count": len(
                candidates
            ),
            "raw_abrupt_count": len(
                abrupt
            ),
            "merged_event_count": len(
                events
            ),
            "threshold_mps2": float(
                threshold
            ),
            "median_change_rate_mps2": (
                median_rate
            ),
            "robust_sigma_mps2": float(
                robust_sigma
            ),
            "minimum_angle_deg": (
                BOUNCE_MIN_ANGLE_DEG
            ),
            "merge_window_s": (
                BOUNCE_MERGE_WINDOW_S
            ),
        },
    )


def bounce_exclusion_mask(
    times: np.ndarray,
    bounces: list[
        dict[
            str,
            Any,
        ]
    ],
) -> np.ndarray:
    """Mark speed samples close to detected bounce events."""
    excluded = np.zeros(
        len(
            times
        ),
        dtype=bool,
    )

    for bounce in bounces:
        bounce_time = float(
            bounce[
                "time_s"
            ]
        )

        excluded |= (
            np.abs(
                times
                - bounce_time
            )
            <= BOUNCE_SUMMARY_EXCLUSION_S
        )

    return excluded


def summarize(
    values: np.ndarray,
    mask: np.ndarray,
    unit: str,
) -> dict[str, Any]:
    """Summarize finite selected values."""
    selected = np.asarray(
        values,
        dtype=np.float64,
    )[mask]

    selected = selected[
        np.isfinite(
            selected
        )
    ]

    if not len(
        selected
    ):
        return {
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
            "unit": unit,
            "samples": 0,
        }

    return {
        "min": float(
            np.min(
                selected
            )
        ),
        "max": float(
            np.max(
                selected
            )
        ),
        "avg": float(
            np.mean(
                selected
            )
        ),
        "unit": unit,
        "samples": int(
            len(
                selected
            )
        ),
    }


def graph_points(
    times: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    bounce_excluded: Optional[
        np.ndarray
    ] = None,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Build graph points from proper samples."""
    points = []

    for index, (
        time_value,
        value,
        keep,
    ) in enumerate(
        zip(
            times,
            values,
            mask,
        )
    ):
        numeric_value = float(
            value
        )

        if (
            not keep
            or not math.isfinite(
                numeric_value
            )
        ):
            continue

        point: dict[
            str,
            Any,
        ] = {
            "t": round(
                float(
                    time_value
                ),
                5,
            ),
            "v": numeric_value,
        }

        if bounce_excluded is not None:
            point[
                "excluded_from_summary"
            ] = bool(
                bounce_excluded[
                    index
                ]
            )

        points.append(
            point
        )

    return points


def analyze_video(
    video_path: str,
    focal_length_px: float,
    target_hsv: tuple[
        int,
        int,
        int,
    ],
    ball_diameter_m: float,
    hue_tolerance_percent: float,
    saturation_tolerance_percent: float,
    value_tolerance_percent: float,
    requested_start_time_s: Optional[float],
    requested_end_time_s: Optional[float],
) -> dict[str, Any]:
    """Calculate speed, acceleration, swing, and bounce events in one window."""
    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():
        raise AppError(
            "The video could not be opened.",
            400,
        )

    try:
        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        if (
            not math.isfinite(
                fps
            )
            or fps <= 0
        ):
            fps = 30.0

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        reported_frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if width <= 0 or height <= 0:
            raise AppError(
                "The video has invalid dimensions.",
                400,
            )

        duration_s: Optional[
            float
        ] = None

        if reported_frame_count > 0:
            duration_s = (
                reported_frame_count
                / fps
            )

        start_time_s = (
            0.0
            if requested_start_time_s
            is None
            else requested_start_time_s
        )

        if requested_end_time_s is None:
            end_time_s = (
                duration_s
                if duration_s is not None
                else math.inf
            )
        else:
            end_time_s = (
                requested_end_time_s
            )

        if start_time_s < 0.0:
            raise AppError(
                "Analysis start time cannot be negative.",
                400,
            )

        if duration_s is not None:
            start_time_s = min(
                start_time_s,
                duration_s,
            )

            end_time_s = min(
                end_time_s,
                duration_s,
            )

        if (
            not math.isfinite(
                start_time_s
            )
            or start_time_s < 0.0
        ):
            raise AppError(
                "Analysis start time is invalid.",
                400,
            )

        if (
            math.isfinite(
                end_time_s
            )
            and end_time_s
            <= start_time_s
        ):
            raise AppError(
                "Analysis end time must be greater than the start time.",
                400,
            )

        start_frame = max(
            0,
            int(
                math.ceil(
                    start_time_s
                    * fps
                    - EPSILON
                )
            ),
        )

        if reported_frame_count > 0:
            start_frame = min(
                start_frame,
                max(
                    reported_frame_count
                    - 1,
                    0,
                ),
            )

        if math.isfinite(
            end_time_s
        ):
            end_frame_inclusive = int(
                math.floor(
                    end_time_s
                    * fps
                    + EPSILON
                )
            )

            if reported_frame_count > 0:
                end_frame_inclusive = min(
                    end_frame_inclusive,
                    reported_frame_count
                    - 1,
                )
        else:
            end_frame_inclusive = None

        if (
            end_frame_inclusive
            is not None
            and end_frame_inclusive
            < start_frame
        ):
            raise AppError(
                "The selected analysis window contains no video frames.",
                400,
            )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame,
        )

        tracker = BallTracker(
            target_hsv,
            hue_tolerance_percent,
            saturation_tolerance_percent,
            value_tolerance_percent,
        )

        detections: list[
            BallDetection
        ] = []

        frame_index = start_frame
        frames_read = 0

        while True:
            if (
                end_frame_inclusive
                is not None
                and frame_index
                > end_frame_inclusive
            ):
                break

            success, frame = (
                capture.read()
            )

            if not success:
                break

            frames_read += 1

            detection = tracker.update(
                frame
            )

            if detection is not None:
                (
                    center_x,
                    center_y,
                    radius,
                ) = detection

                detections.append(
                    BallDetection(
                        frame_index=(
                            frame_index
                        ),
                        time_s=(
                            frame_index
                            / fps
                        ),
                        center_x=float(
                            center_x
                        ),
                        center_y=float(
                            center_y
                        ),
                        radius_px=float(
                            radius
                        ),
                    )
                )

            frame_index += 1

        if frames_read <= 0:
            raise AppError(
                "The selected analysis window contains no readable frames.",
                400,
            )

        actual_window_end_s = (
            (
                frame_index
                - 1
            )
            / fps
        )

        if len(
            detections
        ) < 2:
            raise AppError(
                "The ball was not detected often enough in the selected "
                "analysis window.",
                details={
                    "analysis_window": {
                        "start_s": (
                            start_time_s
                        ),
                        "end_s": (
                            actual_window_end_s
                        ),
                    },
                    "tracker": (
                        tracker.debug_summary()
                    ),
                },
            )

        (
            times,
            positions,
            radii,
        ) = metric_positions(
            detections,
            focal_length_px,
            ball_diameter_m
            / 2.0,
            width
            / 2.0,
            height
            / 2.0,
        )

        velocity = differentiate(
            times,
            positions,
            span=(
                DERIVATIVE_SPAN_FRAMES
            ),
        )

        acceleration = differentiate(
            times,
            velocity.values,
            span=(
                DERIVATIVE_SPAN_FRAMES
            ),
        )

        bounce_velocity = differentiate(
            times,
            positions,
            span=1,
        )

        proper_velocity = (
            proper_derivative_mask(
                velocity
            )
        )

        proper_acceleration = (
            proper_derivative_mask(
                acceleration
            )
        )

        proper_bounce_velocity = (
            proper_derivative_mask(
                bounce_velocity
            )
        )

        radius_ok = radius_quality_mask(
            radii,
            span=(
                DERIVATIVE_SPAN_FRAMES
            ),
        )

        bounce_radius_ok = radius_quality_mask(
            radii,
            span=1,
        )

        speed_graph_mask = (
            proper_velocity
            & radius_ok
        )

        acceleration_mask = (
            proper_velocity
            & proper_acceleration
            & radius_ok
        )

        bounce_velocity_mask = (
            proper_bounce_velocity
            & bounce_radius_ok
        )

        speed_mps = np.linalg.norm(
            velocity.values,
            axis=1,
        )

        speed_kmph = (
            speed_mps
            * 3.6
        )

        acceleration_mps2 = np.linalg.norm(
            acceleration.values,
            axis=1,
        )

        (
            swing_deg_s,
            swing_mask,
        ) = swing_rate(
            times,
            velocity,
            speed_graph_mask,
        )

        (
            bounces,
            bounce_debug,
        ) = detect_bounce_events(
            times,
            bounce_velocity,
            bounce_velocity_mask,
            detections,
        )

        bounce_excluded = (
            bounce_exclusion_mask(
                times,
                bounces,
            )
        )

        speed_summary_mask = (
            speed_graph_mask
            & ~bounce_excluded
        )

        warnings = []

        detection_rate = (
            len(
                detections
            )
            / max(
                frames_read,
                1,
            )
        )

        if detection_rate < 0.55:
            warnings.append(
                "Tracking was intermittent inside the selected window. "
                "Improve lighting or ball/background contrast."
            )

        if (
            int(
                np.count_nonzero(
                    speed_summary_mask
                )
            )
            < 3
        ):
            warnings.append(
                "Only a few non-bounce speed samples passed the "
                "strict derivative/radius checks."
            )

        if bounces:
            warnings.append(
                f"{len(bounces)} bounce/abrupt velocity-change "
                "event(s) detected. Nearby velocities were excluded "
                "from speed min/max/average."
            )

        logger.info(
            "analysis window=%.3f..%.3f frames=%d detections=%d "
            "bounces=%d speed_graph=%d speed_summary=%d",
            start_time_s,
            actual_window_end_s,
            frames_read,
            len(
                detections
            ),
            len(
                bounces
            ),
            int(
                np.count_nonzero(
                    speed_graph_mask
                )
            ),
            int(
                np.count_nonzero(
                    speed_summary_mask
                )
            ),
        )

        return {
            "summary": {
                "speed": summarize(
                    speed_kmph,
                    speed_summary_mask,
                    "km/h",
                ),
                "acceleration": summarize(
                    acceleration_mps2,
                    acceleration_mask,
                    "m/s²",
                ),
                "swing": summarize(
                    swing_deg_s,
                    swing_mask,
                    "°/s",
                ),
            },
            "series": {
                "speed": graph_points(
                    times,
                    speed_kmph,
                    speed_graph_mask,
                    bounce_excluded=(
                        bounce_excluded
                    ),
                ),
                "acceleration": graph_points(
                    times,
                    acceleration_mps2,
                    acceleration_mask,
                ),
                "swing": graph_points(
                    times,
                    swing_deg_s,
                    swing_mask,
                ),
            },
            "bounces": bounces,
            "calibration": {
                "focal_length_px": float(
                    focal_length_px
                ),
                "ball_diameter_cm": float(
                    ball_diameter_m
                    * 100.0
                ),
                "target_hsv": [
                    int(
                        value
                    )
                    for value in target_hsv
                ],
                "tolerances_percent": {
                    "h": float(
                        hue_tolerance_percent
                    ),
                    "s": float(
                        saturation_tolerance_percent
                    ),
                    "v": float(
                        value_tolerance_percent
                    ),
                },
            },
            "video": {
                "fps": fps,
                "width": width,
                "height": height,
                "reported_frame_count": (
                    reported_frame_count
                ),
                "duration_s": (
                    duration_s
                ),
                "analysis_start_s": (
                    start_time_s
                ),
                "analysis_end_s": (
                    actual_window_end_s
                ),
                "frames_read": (
                    frames_read
                ),
                "detections": len(
                    detections
                ),
                "detection_rate": (
                    detection_rate
                ),
            },
            "quality": {
                "proper_velocity_frames": int(
                    np.count_nonzero(
                        proper_velocity
                    )
                ),
                "proper_acceleration_frames": int(
                    np.count_nonzero(
                        proper_acceleration
                    )
                ),
                "speed_graph_frames": int(
                    np.count_nonzero(
                        speed_graph_mask
                    )
                ),
                "speed_summary_frames": int(
                    np.count_nonzero(
                        speed_summary_mask
                    )
                ),
                "speed_samples_excluded_near_bounces": int(
                    np.count_nonzero(
                        speed_graph_mask
                        & bounce_excluded
                    )
                ),
                "acceleration_summary_frames": int(
                    np.count_nonzero(
                        acceleration_mask
                    )
                ),
                "swing_summary_frames": int(
                    np.count_nonzero(
                        swing_mask
                    )
                ),
                "bounce_summary_exclusion_s": (
                    BOUNCE_SUMMARY_EXCLUSION_S
                ),
            },
            "bounce_detection": (
                bounce_debug
            ),
            "tracker": (
                tracker.debug_summary()
            ),
            "warnings": warnings,
            "definitions": {
                "swing": (
                    "3-D velocity direction-change rate "
                    "in degrees per second."
                ),
                "bounce": (
                    "A robust outlier in finite-difference velocity-vector "
                    "change. No parabolic or trajectory fit is used."
                ),
                "speed_summary": (
                    "Speed min/max/average omit samples within "
                    f"±{BOUNCE_SUMMARY_EXCLUSION_S:.2f} s of detected bounces."
                ),
            },
        }

    finally:
        capture.release()


@app.errorhandler(
    RequestEntityTooLarge
)
def handle_large_upload(
    _: RequestEntityTooLarge,
):
    """Return JSON for oversized uploads."""
    return jsonify(
        {
            "error": (
                "That file is too large. "
                "Trim the clip or upload a smaller file."
            )
        }
    ), 413


@app.errorhandler(
    AppError
)
def handle_app_error(
    error: AppError,
):
    """Return readable application errors."""
    logger.warning(
        "AppError: %s",
        error,
    )

    return jsonify(
        {
            "error": str(
                error
            ),
            "details": (
                error.details
            ),
        }
    ), error.status_code


@app.errorhandler(
    Exception
)
def handle_unexpected_error(
    error: Exception,
):
    """Return a production-safe unexpected error."""
    logger.exception(
        "Unexpected failure"
    )

    return jsonify(
        {
            "error": (
                "Speedbowled hit an unexpected processing error. "
                "Try a shorter, clearer clip."
            ),
            "details": (
                str(
                    error
                )
                if app.debug
                else {}
            ),
        }
    ), 500


@app.route("/")
def index():
    """Render Speedbowled."""
    return render_template(
        "index.html",
        default_ball_diameter_cm=(
            DEFAULT_BALL_DIAMETER_CM
        ),
        default_hue_tolerance_percent=(
            DEFAULT_HUE_TOLERANCE_PERCENT
        ),
        default_saturation_tolerance_percent=(
            DEFAULT_SATURATION_TOLERANCE_PERCENT
        ),
        default_value_tolerance_percent=(
            DEFAULT_VALUE_TOLERANCE_PERCENT
        ),
    )


@app.route(
    "/api/health"
)
def health():
    """Render health endpoint."""
    return jsonify(
        {
            "status": "ok",
            "app": "Speedbowled",
        }
    )


@app.route(
    "/api/calibrate",
    methods=["POST"],
)
def calibrate():
    """Calibrate focal length and ball color."""
    image_file = request.files.get(
        "image"
    )

    if image_file is None:
        raise AppError(
            "Capture or upload a calibration image.",
            400,
        )

    payload = image_file.read()

    image = cv2.imdecode(
        np.frombuffer(
            payload,
            dtype=np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise AppError(
            "The calibration image could not be decoded.",
            400,
        )

    x = int(
        round(
            parse_float_field(
                "x",
                0,
                max(
                    image.shape[
                        1
                    ]
                    - 1,
                    0,
                ),
            )
        )
    )

    y = int(
        round(
            parse_float_field(
                "y",
                0,
                max(
                    image.shape[
                        0
                    ]
                    - 1,
                    0,
                ),
            )
        )
    )

    ball_diameter_cm = (
        parse_float_field(
            "ball_diameter_cm",
            1.0,
            50.0,
        )
    )

    hue_tolerance_percent = (
        parse_float_field(
            "hue_tolerance_percent",
            0.0,
            100.0,
        )
    )

    saturation_tolerance_percent = (
        parse_float_field(
            "saturation_tolerance_percent",
            0.0,
            100.0,
        )
    )

    value_tolerance_percent = (
        parse_float_field(
            "value_tolerance_percent",
            0.0,
            100.0,
        )
    )

    result = calibrate_image(
        image,
        x,
        y,
        ball_diameter_cm
        / 100.0,
        hue_tolerance_percent,
        saturation_tolerance_percent,
        value_tolerance_percent,
    )

    result[
        "ball_diameter_cm"
    ] = (
        ball_diameter_cm
    )

    return jsonify(
        result
    )


@app.route(
    "/api/analyze",
    methods=["POST"],
)
def analyze():
    """Analyze an uploaded or browser-recorded bowling video."""
    video_file = request.files.get(
        "video"
    )

    if video_file is None:
        raise AppError(
            "Record or upload a bowling video.",
            400,
        )

    focal_length_px = (
        parse_float_field(
            "focal_length_px",
            10.0,
            50000.0,
        )
    )

    ball_diameter_cm = (
        parse_float_field(
            "ball_diameter_cm",
            1.0,
            50.0,
        )
    )

    hue = int(
        round(
            parse_float_field(
                "hue",
                0,
                179,
            )
        )
    )

    saturation = int(
        round(
            parse_float_field(
                "saturation",
                0,
                255,
            )
        )
    )

    value = int(
        round(
            parse_float_field(
                "value",
                0,
                255,
            )
        )
    )

    hue_tolerance_percent = (
        parse_float_field(
            "hue_tolerance_percent",
            0.0,
            100.0,
        )
    )

    saturation_tolerance_percent = (
        parse_float_field(
            "saturation_tolerance_percent",
            0.0,
            100.0,
        )
    )

    value_tolerance_percent = (
        parse_float_field(
            "value_tolerance_percent",
            0.0,
            100.0,
        )
    )

    start_time_raw = request.form.get(
        "analysis_start_s"
    )

    end_time_raw = request.form.get(
        "analysis_end_s"
    )

    try:
        requested_start_time_s = (
            None
            if start_time_raw
            in (
                None,
                "",
            )
            else float(
                start_time_raw
            )
        )

        requested_end_time_s = (
            None
            if end_time_raw
            in (
                None,
                "",
            )
            else float(
                end_time_raw
            )
        )
    except ValueError as exc:
        raise AppError(
            "Analysis start/end times must be numeric.",
            400,
        ) from exc

    extension = os.path.splitext(
        video_file.filename
        or ""
    )[1].lower()

    if extension not in {
        ".mp4",
        ".mov",
        ".m4v",
        ".webm",
        ".avi",
        ".mkv",
    }:
        extension = ".mp4"

    path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temporary:
            path = (
                temporary.name
            )

            video_file.save(
                path
            )

        return jsonify(
            analyze_video(
                path,
                focal_length_px,
                (
                    hue,
                    saturation,
                    value,
                ),
                ball_diameter_cm
                / 100.0,
                hue_tolerance_percent,
                saturation_tolerance_percent,
                value_tolerance_percent,
                requested_start_time_s,
                requested_end_time_s,
            )
        )

    finally:
        if (
            path
            and os.path.exists(
                path
            )
        ):
            os.remove(
                path
            )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                "5000",
            )
        ),
        debug=(
            os.environ.get(
                "FLASK_DEBUG",
                "0",
            )
            == "1"
        ),
    )
