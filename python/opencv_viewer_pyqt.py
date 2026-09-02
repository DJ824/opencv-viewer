from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class OperationId(Enum):
    ORIGINAL = auto()
    GRAYSCALE = auto()
    GAUSSIAN_BLUR = auto()
    MEDIAN_BLUR = auto()
    EQUALIZE_HISTOGRAM = auto()
    CLAHE = auto()
    NORMALIZE_INTENSITY = auto()
    BRIGHTNESS_CONTRAST = auto()
    GAMMA_CORRECTION = auto()
    ILLUMINATION_CORRECTION = auto()
    DIFFERENCE_OF_GAUSSIAN = auto()
    HSV_MASK = auto()
    BRIGHT_MASK = auto()
    ADAPTIVE_BRIGHT_MASK = auto()
    BRIGHT_COMPONENTS = auto()
    CIRCULAR_BRIGHT_BLOBS = auto()
    BEST_BRIGHT_CANDIDATE = auto()
    SIMPLE_BLOB_DETECTOR = auto()
    MSER_REGIONS = auto()
    HOUGH_CIRCLES = auto()
    CANNY = auto()
    SOBEL_MAGNITUDE = auto()
    LAPLACIAN_EDGES = auto()
    BINARY_THRESHOLD = auto()
    ADAPTIVE_THRESHOLD = auto()
    ERODE = auto()
    DILATE = auto()
    MORPH_OPEN = auto()
    MORPH_CLOSE = auto()
    WHITE_TOP_HAT = auto()
    BLACK_HAT = auto()
    CONTOURS = auto()
    CONNECTED_COMPONENTS = auto()
    FAST_CORNERS = auto()
    ORB_KEYPOINTS = auto()
    BACKGROUND_SUBTRACTION = auto()
    RUNNING_AVERAGE_FOREGROUND = auto()
    MOTION_BRIGHT_MASK = auto()
    FRAME_DIFFERENCE = auto()
    SPARSE_OPTICAL_FLOW = auto()
    OPTICAL_FLOW = auto()
    GOOD_FEATURES = auto()


@dataclass(frozen=True)
class OperationInfo:
    op_id: OperationId
    category: str
    name: str
    description: str
    temporal: bool = False


@dataclass(frozen=True)
class ParameterSpec:
    label: str
    minimum: int
    maximum: int
    default: int


@dataclass
class ParameterWidgets:
    row: QWidget
    label: QLabel
    slider: QSlider
    spin_box: QSpinBox


@dataclass
class PipelineStep:
    op_id: OperationId = OperationId.ORIGINAL
    parameters: list[int] = field(default_factory=list)
    previous_gray: np.ndarray | None = None
    running_average: np.ndarray | None = None
    previous_points: np.ndarray | None = None
    background_subtractor: cv2.BackgroundSubtractorMOG2 | None = None
    background_history: int = 0
    background_threshold: int = 0


def odd_kernel(value: int, minimum: int, maximum: int) -> int:
    value = max(minimum, min(maximum, int(value)))
    if value % 2 == 0:
        value += 1
    if value > maximum:
        value -= 2
    return max(minimum, value)


def ensure_uint8(image: np.ndarray | None) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image / 256).astype(np.uint8)
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def to_gray(image: np.ndarray | None) -> np.ndarray:
    source = ensure_uint8(image)
    if source is None:
        return np.empty((0, 0), dtype=np.uint8)
    if source.ndim == 2:
        return source.copy()
    if source.shape[2] == 3:
        return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    if source.shape[2] == 4:
        return cv2.cvtColor(source, cv2.COLOR_BGRA2GRAY)
    return source[:, :, 0].copy()


def to_bgr(image: np.ndarray | None) -> np.ndarray:
    source = ensure_uint8(image)
    if source is None:
        return np.empty((0, 0, 3), dtype=np.uint8)
    if source.ndim == 2:
        return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    if source.shape[2] == 3:
        return source.copy()
    if source.shape[2] == 4:
        return cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(source[:, :, 0], cv2.COLOR_GRAY2BGR)


def mat_to_qimage(image: np.ndarray | None) -> QImage:
    source = ensure_uint8(image)
    if source is None or source.size == 0:
        return QImage()

    if source.ndim == 2:
        gray = np.ascontiguousarray(source)
        qimage = QImage(
            gray.data,
            gray.shape[1],
            gray.shape[0],
            gray.strides[0],
            QImage.Format.Format_Grayscale8,
        )
        return qimage.copy()

    if source.shape[2] == 3:
        rgb = np.ascontiguousarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        return qimage.copy()

    if source.shape[2] == 4:
        rgba = np.ascontiguousarray(cv2.cvtColor(source, cv2.COLOR_BGRA2RGBA))
        qimage = QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        return qimage.copy()

    return mat_to_qimage(to_gray(source))


def read_image(path: str) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)


def draw_waiting_label(image: np.ndarray, text: str) -> None:
    cv2.putText(image, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 1, cv2.LINE_AA)


def binary_mask(gray: np.ndarray, threshold: int, invert: bool = False) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, threshold, 255, mode)
    return mask


def clean_mask(mask: np.ndarray, open_kernel: int, close_kernel: int, dilation: int) -> np.ndarray:
    output = mask.copy()
    if open_kernel > 1:
        size = odd_kernel(open_kernel, 1, 51)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        size = odd_kernel(close_kernel, 1, 51)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        output = cv2.dilate(output, kernel, iterations=dilation)
    return output


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    output = bgr.copy()
    tinted = np.full_like(output, color)
    output[mask > 0] = tinted[mask > 0]
    return cv2.addWeighted(bgr, 1.0 - alpha, output, alpha, 0.0)


def contour_circularity(contour: np.ndarray) -> float:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= sys.float_info.epsilon:
        return 0.0
    return float(4.0 * math.pi * cv2.contourArea(contour) / (perimeter * perimeter))


def draw_filtered_contours(
    output: np.ndarray,
    mask: np.ndarray,
    min_area: float,
    max_area: float,
    min_circularity: float,
    box_color: tuple[int, int, int],
) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or (max_area > 0.0 and area > max_area):
            continue
        if min_circularity > 0.0 and contour_circularity(contour) < min_circularity:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + width, y + height), box_color, 2)
        cv2.drawContours(output, [contour], -1, (255, 190, 40), 1)

        moments = cv2.moments(contour)
        if abs(moments["m00"]) > sys.float_info.epsilon:
            center = (int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"])))
            cv2.drawMarker(output, center, (20, 40, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)


def operations() -> list[OperationInfo]:
    return [
        OperationInfo(OperationId.ORIGINAL, "Fundamentals", "Original", "Pass the input through unchanged."),
        OperationInfo(OperationId.GRAYSCALE, "Fundamentals", "Grayscale", "Convert the input to a single luminance channel."),
        OperationInfo(OperationId.GAUSSIAN_BLUR, "Fundamentals", "Gaussian blur", "Smooth noise with a Gaussian kernel before segmentation or edge detection."),
        OperationInfo(OperationId.MEDIAN_BLUR, "Fundamentals", "Median blur", "Reduce salt-and-pepper noise while preserving stronger edges."),
        OperationInfo(OperationId.EQUALIZE_HISTOGRAM, "Contrast and illumination", "Equalize histogram", "Spread grayscale contrast across the available intensity range."),
        OperationInfo(OperationId.CLAHE, "Contrast and illumination", "CLAHE", "Boost local contrast while limiting noise amplification."),
        OperationInfo(OperationId.NORMALIZE_INTENSITY, "Contrast and illumination", "Normalize intensity", "Normalize luminance to the full 8-bit range."),
        OperationInfo(OperationId.BRIGHTNESS_CONTRAST, "Contrast and illumination", "Brightness and contrast", "Apply a linear contrast and brightness adjustment."),
        OperationInfo(OperationId.GAMMA_CORRECTION, "Contrast and illumination", "Gamma correction", "Nonlinear intensity remapping for lifting or suppressing bright regions."),
        OperationInfo(OperationId.ILLUMINATION_CORRECTION, "Contrast and illumination", "Divide by local background", "Suppress slow illumination changes by dividing by a blurred background estimate."),
        OperationInfo(OperationId.DIFFERENCE_OF_GAUSSIAN, "Contrast and illumination", "Difference of Gaussian", "Highlight structures that survive a small blur but not a larger background blur."),
        OperationInfo(OperationId.HSV_MASK, "Color and masks", "HSV color mask", "Isolate pixels inside a hue, saturation, and value range."),
        OperationInfo(OperationId.BRIGHT_MASK, "Bright object isolation", "Bright mask", "Keep pixels above a brightness threshold, optionally suppressing highly saturated clutter."),
        OperationInfo(OperationId.ADAPTIVE_BRIGHT_MASK, "Bright object isolation", "Adaptive bright mask", "Extract pixels brighter than their local background."),
        OperationInfo(OperationId.BRIGHT_COMPONENTS, "Bright object isolation", "Bright components", "Threshold bright regions and draw bounding boxes around connected candidates."),
        OperationInfo(OperationId.CIRCULAR_BRIGHT_BLOBS, "Bright object isolation", "Circular bright blobs", "Find bright components that satisfy area and circularity constraints."),
        OperationInfo(OperationId.BEST_BRIGHT_CANDIDATE, "Bright object isolation", "Best bright candidate", "Score bright components by intensity, area, and circularity, then draw the strongest candidate."),
        OperationInfo(OperationId.SIMPLE_BLOB_DETECTOR, "Bright object isolation", "Simple blob detector", "Use OpenCV's blob detector to find bright blob-like regions."),
        OperationInfo(OperationId.MSER_REGIONS, "Bright object isolation", "MSER bright regions", "Find stable extremal regions and keep bright candidates."),
        OperationInfo(OperationId.HOUGH_CIRCLES, "Bright object isolation", "Hough circles", "Detect circular bright-object candidates after grayscale smoothing."),
        OperationInfo(OperationId.RUNNING_AVERAGE_FOREGROUND, "Bright object isolation", "Running-average foreground", "Detect moving foreground against a running average background model.", True),
        OperationInfo(OperationId.MOTION_BRIGHT_MASK, "Bright object isolation", "Motion-gated bright mask", "Combine frame difference with a brightness threshold to reject static bright clutter.", True),
        OperationInfo(OperationId.CANNY, "Edges and thresholds", "Canny edges", "Extract edges using low and high hysteresis thresholds."),
        OperationInfo(OperationId.SOBEL_MAGNITUDE, "Edges and thresholds", "Sobel magnitude", "Show gradient strength from horizontal and vertical Sobel derivatives."),
        OperationInfo(OperationId.LAPLACIAN_EDGES, "Edges and thresholds", "Laplacian edges", "Highlight second-derivative edge responses."),
        OperationInfo(OperationId.BINARY_THRESHOLD, "Edges and thresholds", "Binary threshold", "Convert luminance to a binary foreground/background mask."),
        OperationInfo(OperationId.ADAPTIVE_THRESHOLD, "Edges and thresholds", "Adaptive threshold", "Build a binary mask with local neighborhood thresholds for uneven lighting."),
        OperationInfo(OperationId.ERODE, "Morphology", "Erode mask", "Shrink foreground regions in a thresholded mask."),
        OperationInfo(OperationId.DILATE, "Morphology", "Dilate mask", "Expand foreground regions in a thresholded mask."),
        OperationInfo(OperationId.MORPH_OPEN, "Morphology", "Open mask", "Remove small blobs using erosion followed by dilation."),
        OperationInfo(OperationId.MORPH_CLOSE, "Morphology", "Close mask", "Fill small gaps using dilation followed by erosion."),
        OperationInfo(OperationId.WHITE_TOP_HAT, "Morphology", "White top-hat", "Highlight small bright structures against uneven bright backgrounds."),
        OperationInfo(OperationId.BLACK_HAT, "Morphology", "Black-hat", "Highlight small dark gaps or shadow structures after morphological closing."),
        OperationInfo(OperationId.CONTOURS, "Tracking helpers", "Contours and boxes", "Find edge contours and draw bounding boxes over likely objects."),
        OperationInfo(OperationId.CONNECTED_COMPONENTS, "Tracking helpers", "Connected components", "Draw boxes and centers for connected regions in a binary mask."),
        OperationInfo(OperationId.FAST_CORNERS, "Tracking helpers", "FAST corners", "Mark FAST keypoints that can be useful for frame-to-frame tracking."),
        OperationInfo(OperationId.ORB_KEYPOINTS, "Tracking helpers", "ORB keypoints", "Detect oriented binary features for matching or tracking."),
        OperationInfo(OperationId.BACKGROUND_SUBTRACTION, "Tracking helpers", "MOG2 background subtraction", "Highlight moving foreground using OpenCV's MOG2 background model.", True),
        OperationInfo(OperationId.FRAME_DIFFERENCE, "Tracking helpers", "Frame difference", "Highlight changes between consecutive frames and draw motion boxes.", True),
        OperationInfo(OperationId.SPARSE_OPTICAL_FLOW, "Tracking helpers", "Sparse LK optical flow", "Track good features between consecutive frames using pyramidal Lucas-Kanade flow.", True),
        OperationInfo(OperationId.OPTICAL_FLOW, "Tracking helpers", "Dense optical flow", "Draw Farneback motion vectors between consecutive frames.", True),
        OperationInfo(OperationId.GOOD_FEATURES, "Tracking helpers", "Trackable corners", "Mark strong corners that are useful as tracking features."),
    ]


OPERATIONS = operations()


def operation_info(op_id: OperationId) -> OperationInfo | None:
    return next((operation for operation in OPERATIONS if operation.op_id == op_id), None)


def parameter_specs(op_id: OperationId) -> list[ParameterSpec]:
    match op_id:
        case OperationId.GAUSSIAN_BLUR:
            return [ParameterSpec("Kernel", 1, 51, 7), ParameterSpec("Sigma x10", 0, 100, 15)]
        case OperationId.MEDIAN_BLUR:
            return [ParameterSpec("Kernel", 1, 51, 5)]
        case OperationId.CLAHE:
            return [ParameterSpec("Clip x10", 1, 100, 20), ParameterSpec("Tile", 2, 32, 8)]
        case OperationId.BRIGHTNESS_CONTRAST:
            return [ParameterSpec("Contrast %", 0, 300, 120), ParameterSpec("Brightness", -100, 100, 0)]
        case OperationId.GAMMA_CORRECTION:
            return [ParameterSpec("Gamma x100", 10, 500, 80)]
        case OperationId.ILLUMINATION_CORRECTION:
            return [ParameterSpec("Bg kernel", 3, 201, 61), ParameterSpec("Gain %", 20, 300, 120)]
        case OperationId.DIFFERENCE_OF_GAUSSIAN:
            return [ParameterSpec("Small", 1, 51, 5), ParameterSpec("Large", 3, 201, 41), ParameterSpec("Gain x10", 1, 100, 20)]
        case OperationId.CANNY:
            return [ParameterSpec("Low", 0, 255, 60), ParameterSpec("High", 0, 255, 160), ParameterSpec("Aperture", 3, 7, 3)]
        case OperationId.SOBEL_MAGNITUDE:
            return [ParameterSpec("Kernel", 1, 7, 3), ParameterSpec("Scale x10", 1, 100, 10)]
        case OperationId.LAPLACIAN_EDGES:
            return [ParameterSpec("Kernel", 1, 31, 3), ParameterSpec("Scale x10", 1, 100, 10)]
        case OperationId.BINARY_THRESHOLD:
            return [ParameterSpec("Threshold", 0, 255, 110), ParameterSpec("Max", 1, 255, 255)]
        case OperationId.ADAPTIVE_THRESHOLD:
            return [ParameterSpec("Block", 3, 99, 21), ParameterSpec("C", -50, 50, 5)]
        case OperationId.HSV_MASK:
            return [
                ParameterSpec("Hue min", 0, 179, 0),
                ParameterSpec("Hue max", 0, 179, 25),
                ParameterSpec("Sat min", 0, 255, 80),
                ParameterSpec("Sat max", 0, 255, 255),
                ParameterSpec("Val min", 0, 255, 80),
                ParameterSpec("Val max", 0, 255, 255),
            ]
        case OperationId.BRIGHT_MASK:
            return [
                ParameterSpec("Value", 0, 255, 220),
                ParameterSpec("Max sat", 0, 255, 255),
                ParameterSpec("Open", 0, 31, 3),
                ParameterSpec("Close", 0, 31, 5),
                ParameterSpec("Dilation", 0, 10, 1),
            ]
        case OperationId.ADAPTIVE_BRIGHT_MASK:
            return [
                ParameterSpec("Bg kernel", 3, 201, 61),
                ParameterSpec("Delta", 0, 120, 22),
                ParameterSpec("Open", 0, 31, 3),
                ParameterSpec("Close", 0, 31, 5),
                ParameterSpec("Dilation", 0, 10, 1),
            ]
        case OperationId.ERODE | OperationId.DILATE | OperationId.MORPH_OPEN | OperationId.MORPH_CLOSE:
            return [ParameterSpec("Threshold", 0, 255, 110), ParameterSpec("Kernel", 1, 31, 5), ParameterSpec("Iterations", 1, 10, 1)]
        case OperationId.WHITE_TOP_HAT | OperationId.BLACK_HAT:
            return [ParameterSpec("Kernel", 1, 101, 31), ParameterSpec("Threshold", 0, 255, 25)]
        case OperationId.CONTOURS:
            return [ParameterSpec("Low", 0, 255, 50), ParameterSpec("High", 0, 255, 150), ParameterSpec("Min area", 0, 20000, 500)]
        case OperationId.CONNECTED_COMPONENTS:
            return [
                ParameterSpec("Threshold", 0, 255, 1),
                ParameterSpec("Min area", 0, 20000, 250),
                ParameterSpec("Max area", 0, 200000, 0),
                ParameterSpec("Min circ %", 0, 100, 0),
                ParameterSpec("Dilation", 0, 10, 0),
            ]
        case OperationId.BRIGHT_COMPONENTS:
            return [
                ParameterSpec("Value", 0, 255, 220),
                ParameterSpec("Min area", 0, 20000, 250),
                ParameterSpec("Max area", 0, 200000, 0),
                ParameterSpec("Open", 0, 31, 3),
                ParameterSpec("Close", 0, 31, 5),
                ParameterSpec("Dilation", 0, 10, 1),
            ]
        case OperationId.CIRCULAR_BRIGHT_BLOBS:
            return [
                ParameterSpec("Value", 0, 255, 220),
                ParameterSpec("Min area", 0, 20000, 120),
                ParameterSpec("Max area", 0, 200000, 0),
                ParameterSpec("Min circ %", 0, 100, 55),
                ParameterSpec("Open", 0, 31, 3),
                ParameterSpec("Close", 0, 31, 5),
            ]
        case OperationId.BEST_BRIGHT_CANDIDATE:
            return [
                ParameterSpec("Value", 0, 255, 220),
                ParameterSpec("Min area", 0, 20000, 120),
                ParameterSpec("Max area", 0, 200000, 0),
                ParameterSpec("Min circ %", 0, 100, 30),
                ParameterSpec("Open", 0, 31, 3),
                ParameterSpec("Close", 0, 31, 5),
            ]
        case OperationId.SIMPLE_BLOB_DETECTOR:
            return [
                ParameterSpec("Value", 0, 255, 210),
                ParameterSpec("Min area", 0, 20000, 80),
                ParameterSpec("Max area", 0, 200000, 0),
                ParameterSpec("Min circ %", 0, 100, 30),
                ParameterSpec("Min inert %", 0, 100, 0),
                ParameterSpec("Min conv %", 0, 100, 0),
            ]
        case OperationId.MSER_REGIONS:
            return [
                ParameterSpec("Delta", 1, 20, 5),
                ParameterSpec("Min area", 0, 20000, 80),
                ParameterSpec("Max area", 1, 200000, 8000),
                ParameterSpec("Min value", 0, 255, 175),
            ]
        case OperationId.FAST_CORNERS:
            return [ParameterSpec("Threshold", 1, 100, 24), ParameterSpec("Nonmax", 0, 1, 1), ParameterSpec("Max pts", 10, 2000, 500)]
        case OperationId.ORB_KEYPOINTS:
            return [ParameterSpec("Max pts", 10, 3000, 600), ParameterSpec("Fast thr", 1, 100, 20)]
        case OperationId.HOUGH_CIRCLES:
            return [
                ParameterSpec("dp x10", 10, 30, 12),
                ParameterSpec("Min dist", 1, 300, 40),
                ParameterSpec("Canny", 1, 255, 120),
                ParameterSpec("Votes", 1, 255, 22),
                ParameterSpec("Min r", 0, 200, 0),
                ParameterSpec("Max r", 0, 300, 0),
            ]
        case OperationId.BACKGROUND_SUBTRACTION:
            return [ParameterSpec("History", 10, 500, 120), ParameterSpec("Variance", 1, 128, 16), ParameterSpec("Min area", 0, 20000, 600)]
        case OperationId.RUNNING_AVERAGE_FOREGROUND:
            return [
                ParameterSpec("Alpha x1000", 1, 500, 40),
                ParameterSpec("Threshold", 0, 255, 28),
                ParameterSpec("Min area", 0, 20000, 400),
                ParameterSpec("Dilation", 0, 10, 2),
            ]
        case OperationId.MOTION_BRIGHT_MASK:
            return [
                ParameterSpec("Motion", 0, 255, 25),
                ParameterSpec("Value", 0, 255, 210),
                ParameterSpec("Min area", 0, 20000, 250),
                ParameterSpec("Dilation", 0, 10, 2),
                ParameterSpec("Open", 0, 31, 3),
                ParameterSpec("Close", 0, 31, 5),
            ]
        case OperationId.FRAME_DIFFERENCE:
            return [ParameterSpec("Threshold", 0, 255, 32), ParameterSpec("Min area", 0, 20000, 450), ParameterSpec("Dilation", 0, 10, 2)]
        case OperationId.SPARSE_OPTICAL_FLOW:
            return [
                ParameterSpec("Max pts", 10, 1000, 200),
                ParameterSpec("Quality x1000", 1, 100, 10),
                ParameterSpec("Distance", 1, 100, 10),
                ParameterSpec("Min move", 0, 50, 1),
            ]
        case OperationId.OPTICAL_FLOW:
            return [ParameterSpec("Grid", 8, 64, 24), ParameterSpec("Gain x10", 1, 80, 18), ParameterSpec("Min mag", 0, 50, 3)]
        case OperationId.GOOD_FEATURES:
            return [ParameterSpec("Max", 10, 500, 120), ParameterSpec("Quality x1000", 1, 100, 10), ParameterSpec("Distance", 1, 100, 12)]
        case _:
            return []


def default_parameters(op_id: OperationId) -> list[int]:
    return [spec.default for spec in parameter_specs(op_id)]


def parameter(parameters: list[int], index: int, fallback: int) -> int:
    if index < 0 or index >= len(parameters):
        return fallback
    return parameters[index]


def parameter_summary(op_id: OperationId, parameters: list[int]) -> str:
    entries = [f"{spec.label} {parameter(parameters, index, spec.default)}" for index, spec in enumerate(parameter_specs(op_id))]
    return ", ".join(entries)


def pipeline_step_title(index: int, step: PipelineStep) -> str:
    info = operation_info(step.op_id)
    name = info.name if info else "Unknown operation"
    summary = parameter_summary(step.op_id, step.parameters)
    if summary:
        return f"{index + 1}. {name} ({summary})"
    return f"{index + 1}. {name}"


class ImageView(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText("No media loaded")
        self.setObjectName("imageView")

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._update_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._image.isNull():
            self.clear()
            self.setText("No media loaded")
            return
        self.setText("")
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


class MainWindow(QMainWindow):
    PARAMETER_COUNT = 6

    def __init__(self) -> None:
        super().__init__()
        self.operation_tree: QTreeWidget | None = None
        self.operation_filter: QLineEdit | None = None
        self.pipeline_list: QListWidget | None = None
        self.media_status_label: QLabel | None = None
        self.operation_description_label: QLabel | None = None
        self.frame_label: QLabel | None = None
        self.parameter_group: QGroupBox | None = None
        self.add_step_button: QPushButton | None = None
        self.remove_step_button: QPushButton | None = None
        self.move_step_up_button: QPushButton | None = None
        self.move_step_down_button: QPushButton | None = None
        self.clear_chain_button: QPushButton | None = None
        self.play_pause_button: QPushButton | None = None
        self.stop_button: QPushButton | None = None
        self.previous_frame_button: QPushButton | None = None
        self.next_frame_button: QPushButton | None = None
        self.timeline_slider: QSlider | None = None
        self.stage_scroll_area: QScrollArea | None = None
        self.stage_container: QWidget | None = None
        self.stage_grid_layout: QGridLayout | None = None

        self.parameter_widgets: list[ParameterWidgets] = []
        self.parameter_values = [0] * self.PARAMETER_COUNT
        self.pipeline: list[PipelineStep] = []
        self.preview_step = PipelineStep()
        self.stage_views: list[ImageView] = []
        self.stage_titles: list[str] = []

        self.capture: cv2.VideoCapture | None = None
        self.current_frame: np.ndarray | None = None
        self.current_path = ""
        self.selected_operation = OperationId.ORIGINAL
        self.selected_step_index = -1
        self.is_video = False
        self.is_playing = False
        self.configuring_parameters = False
        self.updating_pipeline_list = False
        self.frame_index = 0
        self.frame_count = 0
        self.frames_per_second = 30.0

        self.frame_timer = QTimer(self)
        self.frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.frame_timer.timeout.connect(self.advance_video)

        self.build_ui()
        self.build_operation_tree()
        self.update_pipeline_list()
        self.select_operation(OperationId.ORIGINAL)
        self.update_playback_controls()
        self.update_media_status()

    def build_ui(self) -> None:
        self.setWindowTitle("OpenCV Operation Viewer - PyQt")
        self.resize(1280, 820)

        central_widget = QWidget(self)
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        left_panel = QWidget(central_widget)
        left_panel.setObjectName("leftPanel")
        left_panel.setMinimumWidth(320)
        left_panel.setMaximumWidth(420)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        open_button = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Open media", left_panel)
        open_button.setMinimumHeight(36)
        open_button.clicked.connect(self.open_media_dialog)

        self.media_status_label = QLabel("No media loaded", left_panel)
        self.media_status_label.setObjectName("metaLabel")
        self.media_status_label.setWordWrap(True)

        self.operation_filter = QLineEdit(left_panel)
        self.operation_filter.setPlaceholderText("Filter operations")
        self.operation_filter.setClearButtonEnabled(True)
        self.operation_filter.textChanged.connect(self.filter_operations)

        self.operation_tree = QTreeWidget(left_panel)
        self.operation_tree.setHeaderHidden(True)
        self.operation_tree.setRootIsDecorated(True)
        self.operation_tree.setUniformRowHeights(True)
        self.operation_tree.setMinimumHeight(260)
        self.operation_tree.header().setStretchLastSection(True)
        self.operation_tree.itemClicked.connect(self._operation_tree_clicked)
        self.operation_tree.itemDoubleClicked.connect(self._operation_tree_double_clicked)

        chain_title_label = QLabel("Operation chain", left_panel)
        chain_title_label.setObjectName("viewerTitle")

        self.pipeline_list = QListWidget(left_panel)
        self.pipeline_list.setMinimumHeight(140)
        self.pipeline_list.setAlternatingRowColors(True)
        self.pipeline_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.pipeline_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.pipeline_list.currentRowChanged.connect(self.select_pipeline_step)
        self.pipeline_list.customContextMenuRequested.connect(self.show_pipeline_context_menu)

        delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self.pipeline_list)
        delete_shortcut.activated.connect(self.remove_selected_pipeline_step)
        backspace_shortcut = QShortcut(QKeySequence.StandardKey.Backspace, self.pipeline_list)
        backspace_shortcut.activated.connect(self.remove_selected_pipeline_step)

        chain_button_row = QWidget(left_panel)
        chain_button_layout = QHBoxLayout(chain_button_row)
        chain_button_layout.setContentsMargins(0, 0, 0, 0)
        chain_button_layout.setSpacing(6)

        self.add_step_button = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton), "Add", chain_button_row)
        self.remove_step_button = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton), "Remove", chain_button_row)
        self.move_step_up_button = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), "", chain_button_row)
        self.move_step_down_button = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown), "", chain_button_row)
        self.clear_chain_button = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton), "", chain_button_row)

        self.add_step_button.setToolTip("Add selected operation to chain")
        self.remove_step_button.setToolTip("Remove selected chain step")
        self.move_step_up_button.setToolTip("Move selected step up")
        self.move_step_down_button.setToolTip("Move selected step down")
        self.clear_chain_button.setToolTip("Clear operation chain")
        self.move_step_up_button.setFixedWidth(34)
        self.move_step_down_button.setFixedWidth(34)
        self.clear_chain_button.setFixedWidth(34)

        self.add_step_button.clicked.connect(self.add_selected_operation_to_chain)
        self.remove_step_button.clicked.connect(self.remove_selected_pipeline_step)
        self.move_step_up_button.clicked.connect(lambda: self.move_selected_pipeline_step(-1))
        self.move_step_down_button.clicked.connect(lambda: self.move_selected_pipeline_step(1))
        self.clear_chain_button.clicked.connect(self.clear_pipeline)

        chain_button_layout.addWidget(self.add_step_button, 1)
        chain_button_layout.addWidget(self.remove_step_button)
        chain_button_layout.addWidget(self.move_step_up_button)
        chain_button_layout.addWidget(self.move_step_down_button)
        chain_button_layout.addWidget(self.clear_chain_button)

        self.operation_description_label = QLabel(left_panel)
        self.operation_description_label.setObjectName("descriptionLabel")
        self.operation_description_label.setWordWrap(True)

        self.parameter_group = QGroupBox("Parameters", left_panel)
        parameter_layout = QVBoxLayout(self.parameter_group)
        parameter_layout.setContentsMargins(10, 14, 10, 10)
        parameter_layout.setSpacing(8)

        for index in range(self.PARAMETER_COUNT):
            row = QWidget(self.parameter_group)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            label = QLabel(self.parameter_group)
            label.setMinimumWidth(92)
            slider = QSlider(Qt.Orientation.Horizontal, self.parameter_group)
            spin_box = QSpinBox(self.parameter_group)
            spin_box.setMinimumWidth(72)

            row_layout.addWidget(label)
            row_layout.addWidget(slider, 1)
            row_layout.addWidget(spin_box)
            parameter_layout.addWidget(row)

            widgets = ParameterWidgets(row=row, label=label, slider=slider, spin_box=spin_box)
            self.parameter_widgets.append(widgets)
            slider.valueChanged.connect(lambda value, parameter_index=index: self.slider_changed(parameter_index, value))
            spin_box.valueChanged.connect(lambda value, parameter_index=index: self.spin_box_changed(parameter_index, value))

        left_layout.addWidget(open_button)
        left_layout.addWidget(self.media_status_label)
        left_layout.addSpacing(8)
        left_layout.addWidget(self.operation_filter)
        left_layout.addWidget(self.operation_tree, 1)
        left_layout.addWidget(chain_title_label)
        left_layout.addWidget(self.pipeline_list)
        left_layout.addWidget(chain_button_row)
        left_layout.addWidget(self.operation_description_label)
        left_layout.addWidget(self.parameter_group)

        main_panel = QWidget(central_widget)
        main_layout = QVBoxLayout(main_panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        stage_header_label = QLabel("Pipeline stages", main_panel)
        stage_header_label.setObjectName("viewerTitle")

        self.stage_scroll_area = QScrollArea(main_panel)
        self.stage_scroll_area.setObjectName("stageScrollArea")
        self.stage_scroll_area.setWidgetResizable(True)

        self.stage_container = QWidget(self.stage_scroll_area)
        self.stage_container.setObjectName("stageContainer")
        self.stage_grid_layout = QGridLayout(self.stage_container)
        self.stage_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.stage_grid_layout.setHorizontalSpacing(10)
        self.stage_grid_layout.setVerticalSpacing(10)
        self.stage_grid_layout.setColumnStretch(0, 1)
        self.stage_grid_layout.setColumnStretch(1, 1)
        self.stage_scroll_area.setWidget(self.stage_container)

        transport = QWidget(main_panel)
        transport.setObjectName("transport")
        transport_layout = QHBoxLayout(transport)
        transport_layout.setContentsMargins(12, 10, 12, 10)
        transport_layout.setSpacing(8)

        self.previous_frame_button = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward), "", transport)
        self.play_pause_button = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "", transport)
        self.stop_button = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop), "", transport)
        self.next_frame_button = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward), "", transport)

        for button, tooltip in (
            (self.previous_frame_button, "Previous frame"),
            (self.play_pause_button, "Play or pause"),
            (self.stop_button, "Stop and rewind"),
            (self.next_frame_button, "Next frame"),
        ):
            button.setToolTip(tooltip)
            button.setFixedSize(36, 32)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal, transport)
        self.timeline_slider.setEnabled(False)

        self.frame_label = QLabel("Frame -", transport)
        self.frame_label.setObjectName("metaLabel")
        self.frame_label.setMinimumWidth(170)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.previous_frame_button.clicked.connect(lambda: self.step_video(-1))
        self.play_pause_button.clicked.connect(lambda: self.set_playing(not self.is_playing))
        self.stop_button.clicked.connect(lambda: (self.set_playing(False), self.seek_video(0)))
        self.next_frame_button.clicked.connect(lambda: self.step_video(1))
        self.timeline_slider.sliderPressed.connect(lambda: self.set_playing(False))
        self.timeline_slider.sliderReleased.connect(lambda: self.seek_video(self.timeline_slider.value()))
        self.timeline_slider.sliderMoved.connect(self.update_frame_label_for_slider)

        transport_layout.addWidget(self.previous_frame_button)
        transport_layout.addWidget(self.play_pause_button)
        transport_layout.addWidget(self.stop_button)
        transport_layout.addWidget(self.next_frame_button)
        transport_layout.addWidget(self.timeline_slider, 1)
        transport_layout.addWidget(self.frame_label)

        main_layout.addWidget(stage_header_label)
        main_layout.addWidget(self.stage_scroll_area, 1)
        main_layout.addWidget(transport)

        root_layout.addWidget(left_panel)
        root_layout.addWidget(main_panel, 1)
        self.setCentralWidget(central_widget)
        self.apply_styles()

    def apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f4f5f7;
                color: #1f2933;
            }
            QWidget#leftPanel,
            QWidget#transport {
                background: #ffffff;
                border: 1px solid #d9dee7;
                border-radius: 8px;
            }
            QScrollArea#stageScrollArea {
                background: transparent;
                border: 0;
            }
            QWidget#stageContainer {
                background: transparent;
            }
            QWidget#stagePane {
                background: #ffffff;
                border: 1px solid #d9dee7;
                border-radius: 8px;
            }
            QLabel#viewerTitle {
                color: #253041;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#metaLabel {
                color: #637083;
                font-size: 12px;
            }
            QLabel#descriptionLabel {
                color: #364152;
                background: #eef3f8;
                border: 1px solid #d5e0eb;
                border-radius: 6px;
                padding: 8px;
            }
            QLabel#imageView {
                background: #111827;
                border: 1px solid #2f3b4f;
                border-radius: 6px;
                color: #9aa7b8;
            }
            QLineEdit,
            QSpinBox {
                border: 1px solid #cfd7e3;
                border-radius: 6px;
                padding: 5px 7px;
                background: #ffffff;
            }
            QPushButton {
                border: 1px solid #c8d1de;
                border-radius: 6px;
                padding: 6px 10px;
                background: #f8fafc;
            }
            QPushButton:hover {
                background: #eef4fb;
            }
            QPushButton:pressed {
                background: #dfe9f5;
            }
            QPushButton:disabled {
                color: #98a4b3;
                background: #f2f4f7;
            }
            QTreeWidget,
            QListWidget {
                border: 1px solid #d6dde8;
                border-radius: 6px;
                background: #ffffff;
                outline: 0;
            }
            QTreeWidget::item,
            QListWidget::item {
                min-height: 26px;
                padding: 2px 4px;
            }
            QTreeWidget::item:selected,
            QListWidget::item:selected {
                background: #2f6f9f;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #d6dde8;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            """
        )

    def build_operation_tree(self) -> None:
        assert self.operation_tree is not None
        self.operation_tree.clear()
        categories: dict[str, QTreeWidgetItem] = {}

        for operation in OPERATIONS:
            if operation.category not in categories:
                categories[operation.category] = QTreeWidgetItem(self.operation_tree, [operation.category])
                categories[operation.category].setFirstColumnSpanned(True)

            item = QTreeWidgetItem(categories[operation.category], [operation.name])
            item.setData(0, Qt.ItemDataRole.UserRole, operation.op_id)
            item.setToolTip(0, operation.description)

        self.operation_tree.expandAll()
        matches = self.operation_tree.findItems("Original", Qt.MatchFlag.MatchRecursive)
        if matches:
            self.operation_tree.setCurrentItem(matches[0])

    def configure_parameters(self, op_id: OperationId, values: list[int] | None = None) -> None:
        self.configuring_parameters = True
        values = values or []

        for widgets in self.parameter_widgets:
            widgets.row.hide()

        specs = parameter_specs(op_id)
        for index, spec in enumerate(specs[: self.PARAMETER_COUNT]):
            widgets = self.parameter_widgets[index]
            value = parameter(values, index, spec.default)
            value = max(spec.minimum, min(spec.maximum, value))

            widgets.label.setText(spec.label)
            widgets.slider.setRange(spec.minimum, spec.maximum)
            widgets.spin_box.setRange(spec.minimum, spec.maximum)
            widgets.slider.setValue(value)
            widgets.spin_box.setValue(value)
            self.parameter_values[index] = value
            widgets.row.show()

        assert self.parameter_group is not None
        self.parameter_group.setVisible(bool(specs))
        self.configuring_parameters = False

    def filter_operations(self, text: str) -> None:
        assert self.operation_tree is not None
        needle = text.strip().lower()

        for top_index in range(self.operation_tree.topLevelItemCount()):
            category = self.operation_tree.topLevelItem(top_index)
            category_matches = not needle or needle in category.text(0).lower()

            for child_index in range(category.childCount()):
                child = category.child(child_index)
                info = operation_info(child.data(0, Qt.ItemDataRole.UserRole))
                child_matches = (
                    not needle
                    or needle in child.text(0).lower()
                    or bool(info and needle in info.description.lower())
                )
                child.setHidden(not child_matches and not category_matches)
                category_matches = category_matches or child_matches

            category.setHidden(not category_matches)
            if needle:
                category.setExpanded(True)

    def open_media_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image or video",
            "",
            "Media files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.mp4 *.mov *.avi *.mkv *.webm *.m4v);;"
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;"
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;"
            "All files (*)",
        )
        if path:
            self.load_media(path)

    def load_media(self, path: str) -> None:
        self.reset_media()
        self.current_path = path

        image = read_image(path)
        if image is not None and image.size > 0:
            self.current_frame = image
            self.is_video = False
            self.frame_index = 0
            self.frame_count = 1
            self.frames_per_second = 0.0
            assert self.timeline_slider is not None
            self.timeline_slider.setRange(0, 0)
            self.process_current_frame()
            self.update_playback_controls()
            self.update_media_status()
            return

        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            self.reset_media()
            QMessageBox.warning(self, "Open media", "Could not open the selected file as an image or video.")
            return

        self.is_video = True
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frames_per_second = fps if math.isfinite(fps) and fps > 0.0 else 30.0
        frame_count = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = int(frame_count) if math.isfinite(frame_count) and frame_count > 0.0 else 0
        assert self.timeline_slider is not None
        self.timeline_slider.setRange(0, max(0, self.frame_count - 1))

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.reset_media()
            QMessageBox.warning(self, "Open media", "The selected video opened but no frames could be decoded.")
            return

        self.current_frame = frame
        self.frame_index = 0
        self.process_current_frame()
        self.update_playback_controls()
        self.update_media_status()

    def reset_media(self) -> None:
        self.set_playing(False)
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.current_frame = None
        self.current_path = ""
        self.is_video = False
        self.frame_index = 0
        self.frame_count = 0
        self.frames_per_second = 30.0
        if self.timeline_slider is not None:
            self.timeline_slider.setRange(0, 0)
            self.timeline_slider.setValue(0)
        self.reset_temporal_state()

    def reset_temporal_state(self) -> None:
        self.reset_step_state(self.preview_step)
        for step in self.pipeline:
            self.reset_step_state(step)

    def reset_temporal_state_from(self, first_step: int) -> None:
        if first_step < 0:
            self.reset_temporal_state()
            return
        for step in self.pipeline[first_step:]:
            self.reset_step_state(step)

    @staticmethod
    def reset_step_state(step: PipelineStep) -> None:
        step.previous_gray = None
        step.running_average = None
        step.previous_points = None
        step.background_subtractor = None
        step.background_history = 0
        step.background_threshold = 0

    def select_operation(self, op_id: OperationId) -> None:
        self.selected_operation = op_id
        self.selected_step_index = -1
        if self.pipeline_list is not None:
            self.pipeline_list.blockSignals(True)
            self.pipeline_list.setCurrentRow(-1)
            self.pipeline_list.blockSignals(False)

        defaults = default_parameters(op_id)
        self.configure_parameters(op_id, defaults)
        self.preview_step.op_id = op_id
        self.preview_step.parameters = self.parameter_values.copy()
        self.reset_step_state(self.preview_step)

        info = operation_info(op_id)
        if info and self.operation_description_label is not None:
            self.operation_description_label.setText(info.description)

        self.process_current_frame()
        self.update_pipeline_controls()

    def select_pipeline_step(self, row: int) -> None:
        if self.updating_pipeline_list:
            return

        if row < 0 or row >= len(self.pipeline):
            self.selected_step_index = -1
            defaults = default_parameters(self.selected_operation)
            self.configure_parameters(self.selected_operation, defaults)
            self.preview_step.op_id = self.selected_operation
            self.preview_step.parameters = self.parameter_values.copy()
            info = operation_info(self.selected_operation)
            if info and self.operation_description_label is not None:
                self.operation_description_label.setText(info.description)
            self.update_pipeline_controls()
            self.process_current_frame()
            return

        self.selected_step_index = row
        step = self.pipeline[row]
        self.selected_operation = step.op_id
        self.configure_parameters(step.op_id, step.parameters)

        info = operation_info(step.op_id)
        if info and self.operation_description_label is not None:
            summary = parameter_summary(step.op_id, step.parameters)
            text = f"Step {row + 1}: {info.name}\n{info.description}"
            if summary:
                text += f"\n{summary}"
            self.operation_description_label.setText(text)

        self.update_pipeline_controls()
        self.process_current_frame()

    def add_selected_operation_to_chain(self) -> None:
        step = PipelineStep(op_id=self.selected_operation, parameters=self.parameter_values.copy())
        self.pipeline.append(step)
        self.selected_step_index = len(self.pipeline) - 1
        self.update_pipeline_list()
        self.reset_temporal_state_from(self.selected_step_index)
        self.select_pipeline_step(self.selected_step_index)
        self.process_current_frame()

    def remove_selected_pipeline_step(self) -> None:
        if not self.has_selected_pipeline_step():
            return
        removed_index = self.selected_step_index
        del self.pipeline[removed_index]
        self.selected_step_index = -1 if not self.pipeline else min(removed_index, len(self.pipeline) - 1)
        self.update_pipeline_list()
        self.reset_temporal_state_from(removed_index)
        self.select_pipeline_step(self.selected_step_index)
        self.process_current_frame()

    def move_selected_pipeline_step(self, direction: int) -> None:
        if not self.has_selected_pipeline_step() or direction == 0:
            return
        target_index = self.selected_step_index + direction
        if target_index < 0 or target_index >= len(self.pipeline):
            return

        step = self.pipeline.pop(self.selected_step_index)
        self.pipeline.insert(target_index, step)
        reset_from = min(self.selected_step_index, target_index)
        self.selected_step_index = target_index
        self.update_pipeline_list()
        self.reset_temporal_state_from(reset_from)
        self.select_pipeline_step(self.selected_step_index)
        self.process_current_frame()

    def clear_pipeline(self) -> None:
        if not self.pipeline:
            return
        self.pipeline.clear()
        self.selected_step_index = -1
        self.update_pipeline_list()
        self.reset_temporal_state()
        self.select_pipeline_step(-1)
        self.process_current_frame()

    def update_pipeline_list(self) -> None:
        if self.pipeline_list is None:
            return

        self.updating_pipeline_list = True
        self.pipeline_list.blockSignals(True)
        self.pipeline_list.clear()

        for index, step in enumerate(self.pipeline):
            item = QListWidgetItem(pipeline_step_title(index, step), self.pipeline_list)
            info = operation_info(step.op_id)
            if info:
                summary = parameter_summary(step.op_id, step.parameters)
                item.setToolTip(f"{info.description}\n{summary}" if summary else info.description)

        if self.has_selected_pipeline_step():
            self.pipeline_list.setCurrentRow(self.selected_step_index)

        self.pipeline_list.blockSignals(False)
        self.updating_pipeline_list = False
        self.update_pipeline_controls()

    def update_pipeline_controls(self) -> None:
        has_selection = self.has_selected_pipeline_step()
        if self.add_step_button is not None:
            self.add_step_button.setEnabled(True)
        if self.remove_step_button is not None:
            self.remove_step_button.setEnabled(has_selection)
        if self.move_step_up_button is not None:
            self.move_step_up_button.setEnabled(has_selection and self.selected_step_index > 0)
        if self.move_step_down_button is not None:
            self.move_step_down_button.setEnabled(has_selection and self.selected_step_index < len(self.pipeline) - 1)
        if self.clear_chain_button is not None:
            self.clear_chain_button.setEnabled(bool(self.pipeline))

    def store_parameter_value(self, index: int, value: int) -> None:
        if index < 0 or index >= len(self.parameter_values):
            return

        self.parameter_values[index] = value
        if self.has_selected_pipeline_step():
            step = self.pipeline[self.selected_step_index]
            if not step.parameters:
                step.parameters = default_parameters(step.op_id)
            while index >= len(step.parameters):
                step.parameters.append(0)
            step.parameters[index] = value
            self.reset_temporal_state_from(self.selected_step_index)
            self.update_pipeline_list()

            info = operation_info(step.op_id)
            if info and self.operation_description_label is not None:
                summary = parameter_summary(step.op_id, step.parameters)
                text = f"Step {self.selected_step_index + 1}: {info.name}\n{info.description}"
                if summary:
                    text += f"\n{summary}"
                self.operation_description_label.setText(text)
            return

        self.preview_step.parameters = self.parameter_values.copy()
        self.reset_step_state(self.preview_step)

    def process_current_frame(self) -> None:
        if self.current_frame is None:
            self.update_stage_images(["Input"], [None])
            self.update_media_status()
            return

        titles = ["Input"]
        frames: list[np.ndarray | None] = [self.current_frame]

        if not self.pipeline:
            self.preview_step.op_id = self.selected_operation
            self.preview_step.parameters = self.parameter_values.copy()
            info = operation_info(self.selected_operation)
            name = info.name if info else "Preview"
            titles.append(f"Preview - {name}")
            frames.append(self.apply_operation(self.current_frame, self.preview_step))
        else:
            stage = self.current_frame
            for index, step in enumerate(self.pipeline):
                info = operation_info(step.op_id)
                name = info.name if info else "Unknown operation"
                stage = self.apply_operation(stage, step)
                titles.append(f"{index + 1}. {name}")
                frames.append(stage)

        self.update_stage_images(titles, frames)
        self.update_media_status()

    def update_stage_images(self, titles: list[str], frames: list[np.ndarray | None]) -> None:
        self.rebuild_stage_views(titles)
        for index, view in enumerate(self.stage_views):
            frame = frames[index] if index < len(frames) else None
            view.set_image(mat_to_qimage(frame))

    def rebuild_stage_views(self, titles: list[str]) -> None:
        if titles == self.stage_titles:
            return
        assert self.stage_grid_layout is not None
        assert self.stage_container is not None

        while self.stage_grid_layout.count():
            item = self.stage_grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.stage_views.clear()
        self.stage_titles = titles.copy()

        for index, title_text in enumerate(titles):
            pane = QWidget(self.stage_container)
            pane.setObjectName("stagePane")
            layout = QVBoxLayout(pane)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(7)

            title = QLabel(title_text, pane)
            title.setObjectName("viewerTitle")
            view = ImageView(pane)
            layout.addWidget(title)
            layout.addWidget(view, 1)

            self.stage_grid_layout.addWidget(pane, index // 2, index % 2)
            self.stage_views.append(view)

        self.stage_grid_layout.setColumnStretch(0, 1)
        self.stage_grid_layout.setColumnStretch(1, 1)

    def update_media_status(self) -> None:
        assert self.media_status_label is not None
        assert self.frame_label is not None
        if self.current_frame is None:
            self.media_status_label.setText("No media loaded")
            self.frame_label.setText("Frame -")
            self.setWindowTitle("OpenCV Operation Viewer - PyQt")
            return

        filename = Path(self.current_path).name
        height, width = self.current_frame.shape[:2]
        dimensions = f"{width} x {height}"

        if self.is_video:
            frame_text = f"Frame {self.frame_index + 1} / {self.frame_count}" if self.frame_count > 0 else f"Frame {self.frame_index + 1}"
            self.media_status_label.setText(f"Video: {filename}\n{dimensions}, {self.frames_per_second:.1f} fps")
            self.frame_label.setText(frame_text)
        else:
            self.media_status_label.setText(f"Image: {filename}\n{dimensions}")
            self.frame_label.setText("Still image")

        if self.timeline_slider is not None:
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(self.frame_index)
            self.timeline_slider.blockSignals(False)
        self.setWindowTitle(f"{filename} - OpenCV Operation Viewer - PyQt")

    def update_playback_controls(self) -> None:
        can_scrub = self.is_video and self.current_frame is not None
        for button in (self.previous_frame_button, self.play_pause_button, self.stop_button, self.next_frame_button):
            if button is not None:
                button.setEnabled(can_scrub)
        if self.timeline_slider is not None:
            self.timeline_slider.setEnabled(can_scrub and self.frame_count > 1)
        if self.play_pause_button is not None:
            icon = QStyle.StandardPixmap.SP_MediaPause if self.is_playing else QStyle.StandardPixmap.SP_MediaPlay
            self.play_pause_button.setIcon(self.style().standardIcon(icon))

    def seek_video(self, frame_number: int) -> None:
        if not self.is_video or self.capture is None or not self.capture.isOpened():
            return

        frame_number = max(0, int(frame_number))
        if self.frame_count > 0:
            frame_number = min(frame_number, self.frame_count - 1)

        self.set_playing(False)
        self.reset_temporal_state()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return

        self.current_frame = frame
        self.frame_index = frame_number
        self.process_current_frame()
        self.update_playback_controls()

    def advance_video(self) -> None:
        if not self.is_video or self.capture is None or not self.capture.isOpened():
            self.set_playing(False)
            return

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.set_playing(False)
            return

        self.current_frame = frame
        next_frame = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        self.frame_index = int(next_frame) - 1 if math.isfinite(next_frame) and next_frame > 0.0 else self.frame_index + 1
        self.process_current_frame()
        self.update_playback_controls()

    def set_playing(self, playing: bool) -> None:
        if playing and (not self.is_video or self.current_frame is None):
            playing = False
        self.is_playing = playing
        if self.is_playing:
            interval_ms = max(1, int(round(1000.0 / self.frames_per_second)))
            self.frame_timer.start(interval_ms)
        else:
            self.frame_timer.stop()
        self.update_playback_controls()

    def step_video(self, frame_delta: int) -> None:
        if self.is_video:
            self.seek_video(self.frame_index + frame_delta)

    def update_frame_label_for_slider(self, value: int) -> None:
        if self.frame_count > 0 and self.frame_label is not None:
            self.frame_label.setText(f"Frame {value + 1} / {self.frame_count}")

    def slider_changed(self, index: int, value: int) -> None:
        if self.configuring_parameters:
            return
        widgets = self.parameter_widgets[index]
        widgets.spin_box.blockSignals(True)
        widgets.spin_box.setValue(value)
        widgets.spin_box.blockSignals(False)
        self.store_parameter_value(index, value)
        self.process_current_frame()

    def spin_box_changed(self, index: int, value: int) -> None:
        if self.configuring_parameters:
            return
        widgets = self.parameter_widgets[index]
        widgets.slider.blockSignals(True)
        widgets.slider.setValue(value)
        widgets.slider.blockSignals(False)
        self.store_parameter_value(index, value)
        self.process_current_frame()

    def show_pipeline_context_menu(self, position) -> None:
        assert self.pipeline_list is not None
        index = self.pipeline_list.indexAt(position)
        if not index.isValid():
            return
        self.pipeline_list.setCurrentRow(index.row())
        menu = QMenu(self)
        remove_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton), "Remove selected operation", self)
        remove_action.triggered.connect(self.remove_selected_pipeline_step)
        menu.addAction(remove_action)
        menu.exec(self.pipeline_list.viewport().mapToGlobal(position))

    def _operation_tree_clicked(self, item: QTreeWidgetItem) -> None:
        if item is None or item.childCount() > 0:
            return
        self.select_operation(item.data(0, Qt.ItemDataRole.UserRole))

    def _operation_tree_double_clicked(self, item: QTreeWidgetItem) -> None:
        if item is None or item.childCount() > 0:
            return
        self.select_operation(item.data(0, Qt.ItemDataRole.UserRole))
        self.add_selected_operation_to_chain()

    def has_selected_pipeline_step(self) -> bool:
        return 0 <= self.selected_step_index < len(self.pipeline)

    def apply_operation(self, frame: np.ndarray, step: PipelineStep) -> np.ndarray:
        bgr = to_bgr(frame)
        gray = to_gray(frame)
        params = step.parameters
        op_id = step.op_id

        match op_id:
            case OperationId.ORIGINAL:
                return frame.copy()
            case OperationId.GRAYSCALE:
                return gray
            case OperationId.GAUSSIAN_BLUR:
                kernel = odd_kernel(parameter(params, 0, 7), 1, 51)
                sigma = parameter(params, 1, 15) / 10.0
                return cv2.GaussianBlur(bgr, (kernel, kernel), sigma)
            case OperationId.MEDIAN_BLUR:
                kernel = odd_kernel(parameter(params, 0, 5), 1, 51)
                return cv2.medianBlur(bgr, kernel)
            case OperationId.EQUALIZE_HISTOGRAM:
                return cv2.equalizeHist(gray)
            case OperationId.CLAHE:
                clip_limit = parameter(params, 0, 20) / 10.0
                tile_size = max(2, parameter(params, 1, 8))
                return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)).apply(gray)
            case OperationId.NORMALIZE_INTENSITY:
                return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            case OperationId.BRIGHTNESS_CONTRAST:
                contrast = parameter(params, 0, 120) / 100.0
                brightness = parameter(params, 1, 0)
                return cv2.convertScaleAbs(bgr, alpha=contrast, beta=brightness)
            case OperationId.GAMMA_CORRECTION:
                gamma = max(0.1, parameter(params, 0, 80) / 100.0)
                lookup = np.array([np.clip((value / 255.0) ** gamma * 255.0, 0, 255) for value in range(256)], dtype=np.uint8)
                return cv2.LUT(bgr, lookup)
            case OperationId.ILLUMINATION_CORRECTION:
                kernel = odd_kernel(parameter(params, 0, 61), 3, 201)
                gain = parameter(params, 1, 120) / 100.0
                gray_float = gray.astype(np.float32)
                background = cv2.GaussianBlur(gray_float, (kernel, kernel), 0.0)
                corrected = cv2.divide(gray_float, background + 1.0, scale=128.0 * gain)
                return np.clip(corrected, 0, 255).astype(np.uint8)
            case OperationId.DIFFERENCE_OF_GAUSSIAN:
                small_kernel = odd_kernel(parameter(params, 0, 5), 1, 51)
                large_kernel = odd_kernel(max(small_kernel + 2, parameter(params, 1, 41)), 3, 201)
                gain = parameter(params, 2, 20) / 10.0
                small_blur = cv2.GaussianBlur(gray, (small_kernel, small_kernel), 0.0)
                large_blur = cv2.GaussianBlur(gray, (large_kernel, large_kernel), 0.0)
                difference = cv2.subtract(small_blur, large_blur)
                return cv2.convertScaleAbs(difference, alpha=gain)
            case OperationId.HSV_MASK:
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                hue_min = parameter(params, 0, 0)
                hue_max = parameter(params, 1, 25)
                sat_min = min(parameter(params, 2, 80), parameter(params, 3, 255))
                sat_max = max(parameter(params, 2, 80), parameter(params, 3, 255))
                val_min = min(parameter(params, 4, 80), parameter(params, 5, 255))
                val_max = max(parameter(params, 4, 80), parameter(params, 5, 255))
                if hue_min <= hue_max:
                    mask = cv2.inRange(hsv, (hue_min, sat_min, val_min), (hue_max, sat_max, val_max))
                else:
                    lower = cv2.inRange(hsv, (0, sat_min, val_min), (hue_max, sat_max, val_max))
                    upper = cv2.inRange(hsv, (hue_min, sat_min, val_min), (179, sat_max, val_max))
                    mask = cv2.bitwise_or(lower, upper)
                output = np.zeros_like(bgr)
                output[mask > 0] = bgr[mask > 0]
                return output
            case OperationId.BRIGHT_MASK:
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                value_mask = binary_mask(hsv[:, :, 2], parameter(params, 0, 220))
                saturation_mask = binary_mask(hsv[:, :, 1], parameter(params, 1, 255), invert=True)
                mask = cv2.bitwise_and(value_mask, saturation_mask)
                return clean_mask(mask, parameter(params, 2, 3), parameter(params, 3, 5), parameter(params, 4, 1))
            case OperationId.ADAPTIVE_BRIGHT_MASK:
                kernel = odd_kernel(parameter(params, 0, 61), 3, 201)
                gray_float = gray.astype(np.float32)
                background = cv2.GaussianBlur(gray_float, (kernel, kernel), 0.0)
                residual = gray_float - background
                residual = np.clip(residual + 128.0, 0, 255).astype(np.uint8)
                mask = binary_mask(residual, 128 + parameter(params, 1, 22))
                return clean_mask(mask, parameter(params, 2, 3), parameter(params, 3, 5), parameter(params, 4, 1))
            case OperationId.BINARY_THRESHOLD:
                _, output = cv2.threshold(gray, parameter(params, 0, 110), parameter(params, 1, 255), cv2.THRESH_BINARY)
                return output
            case OperationId.ADAPTIVE_THRESHOLD:
                block = odd_kernel(parameter(params, 0, 21), 3, 99)
                return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, parameter(params, 1, 5))
            case OperationId.CANNY:
                low = parameter(params, 0, 60)
                high = max(low + 1, parameter(params, 1, 160))
                aperture = odd_kernel(parameter(params, 2, 3), 3, 7)
                return cv2.Canny(gray, low, high, apertureSize=aperture, L2gradient=True)
            case OperationId.SOBEL_MAGNITUDE:
                kernel = odd_kernel(parameter(params, 0, 3), 1, 7)
                scale = parameter(params, 1, 10) / 10.0
                grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel)
                grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel)
                magnitude = cv2.magnitude(grad_x, grad_y)
                return cv2.convertScaleAbs(magnitude, alpha=scale)
            case OperationId.LAPLACIAN_EDGES:
                kernel = odd_kernel(parameter(params, 0, 3), 1, 31)
                scale = parameter(params, 1, 10) / 10.0
                laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=kernel, scale=scale)
                return cv2.convertScaleAbs(laplacian)
            case OperationId.ERODE | OperationId.DILATE | OperationId.MORPH_OPEN | OperationId.MORPH_CLOSE:
                mask = binary_mask(gray, parameter(params, 0, 110))
                kernel_size = odd_kernel(parameter(params, 1, 5), 1, 31)
                iterations = parameter(params, 2, 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                if op_id == OperationId.ERODE:
                    return cv2.erode(mask, kernel, iterations=iterations)
                if op_id == OperationId.DILATE:
                    return cv2.dilate(mask, kernel, iterations=iterations)
                morph_op = cv2.MORPH_OPEN if op_id == OperationId.MORPH_OPEN else cv2.MORPH_CLOSE
                return cv2.morphologyEx(mask, morph_op, kernel, iterations=iterations)
            case OperationId.WHITE_TOP_HAT | OperationId.BLACK_HAT:
                kernel_size = odd_kernel(parameter(params, 0, 31), 1, 101)
                threshold = parameter(params, 1, 25)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                morph_op = cv2.MORPH_TOPHAT if op_id == OperationId.WHITE_TOP_HAT else cv2.MORPH_BLACKHAT
                response = cv2.morphologyEx(gray, morph_op, kernel)
                return binary_mask(response, threshold) if threshold > 0 else response
            case OperationId.CONTOURS:
                low = parameter(params, 0, 50)
                high = max(low + 1, parameter(params, 1, 150))
                edges = cv2.Canny(gray, low, high)
                output = bgr.copy()
                draw_filtered_contours(output, edges, parameter(params, 2, 500), 0.0, 0.0, (30, 210, 120))
                return output
            case OperationId.CONNECTED_COMPONENTS:
                mask = clean_mask(binary_mask(gray, parameter(params, 0, 1)), 0, 0, parameter(params, 4, 0))
                output = overlay_mask(bgr, mask, (60, 180, 255), 0.28)
                draw_filtered_contours(output, mask, parameter(params, 1, 250), parameter(params, 2, 0), parameter(params, 3, 0) / 100.0, (40, 220, 255))
                return output
            case OperationId.BRIGHT_COMPONENTS:
                mask = clean_mask(binary_mask(gray, parameter(params, 0, 220)), parameter(params, 3, 3), parameter(params, 4, 5), parameter(params, 5, 1))
                output = overlay_mask(bgr, mask, (40, 190, 255), 0.32)
                draw_filtered_contours(output, mask, parameter(params, 1, 250), parameter(params, 2, 0), 0.0, (30, 220, 255))
                return output
            case OperationId.CIRCULAR_BRIGHT_BLOBS:
                mask = clean_mask(binary_mask(gray, parameter(params, 0, 220)), parameter(params, 4, 3), parameter(params, 5, 5), 0)
                output = overlay_mask(bgr, mask, (40, 190, 255), 0.32)
                draw_filtered_contours(output, mask, parameter(params, 1, 120), parameter(params, 2, 0), parameter(params, 3, 55) / 100.0, (30, 220, 255))
                return output
            case OperationId.BEST_BRIGHT_CANDIDATE:
                return self.best_bright_candidate(bgr, gray, params)
            case OperationId.SIMPLE_BLOB_DETECTOR:
                detector_params = cv2.SimpleBlobDetector_Params()
                detector_params.minThreshold = float(min(parameter(params, 0, 210), 254))
                detector_params.maxThreshold = 255.0
                detector_params.thresholdStep = 10.0
                detector_params.minRepeatability = 1
                detector_params.filterByColor = True
                detector_params.blobColor = 255
                detector_params.filterByArea = True
                detector_params.minArea = float(max(1, parameter(params, 1, 80)))
                max_area = float(parameter(params, 2, 0) if parameter(params, 2, 0) > 0 else gray.shape[0] * gray.shape[1])
                detector_params.maxArea = max(max_area, detector_params.minArea + 1.0)

                min_circularity = parameter(params, 3, 30)
                detector_params.filterByCircularity = min_circularity > 0
                if detector_params.filterByCircularity:
                    detector_params.minCircularity = min_circularity / 100.0

                min_inertia = parameter(params, 4, 0)
                detector_params.filterByInertia = min_inertia > 0
                if detector_params.filterByInertia:
                    detector_params.minInertiaRatio = min_inertia / 100.0

                min_convexity = parameter(params, 5, 0)
                detector_params.filterByConvexity = min_convexity > 0
                if detector_params.filterByConvexity:
                    detector_params.minConvexity = min_convexity / 100.0
                keypoints = cv2.SimpleBlobDetector_create(detector_params).detect(gray)
                return cv2.drawKeypoints(bgr, keypoints, None, (30, 220, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            case OperationId.MSER_REGIONS:
                output = bgr.copy()
                delta = parameter(params, 0, 5)
                min_area = max(1, parameter(params, 1, 80))
                max_area = max(min_area + 1, parameter(params, 2, 8000))
                min_value = parameter(params, 3, 175)
                regions, boxes = cv2.MSER_create(delta, min_area, max_area).detectRegions(gray)
                for region, box in zip(regions, boxes, strict=False):
                    x, y, width, height = box
                    if width <= 0 or height <= 0:
                        continue
                    crop = gray[y : y + height, x : x + width]
                    if crop.size == 0 or float(np.mean(crop)) < min_value:
                        continue
                    cv2.rectangle(output, (x, y), (x + width, y + height), (30, 220, 255), 2)
                    cv2.polylines(output, [region.reshape(-1, 1, 2)], True, (255, 190, 40), 1)
                return output
            case OperationId.HOUGH_CIRCLES:
                blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    parameter(params, 0, 12) / 10.0,
                    float(parameter(params, 1, 40)),
                    param1=float(parameter(params, 2, 120)),
                    param2=float(parameter(params, 3, 22)),
                    minRadius=parameter(params, 4, 0),
                    maxRadius=parameter(params, 5, 0),
                )
                output = bgr.copy()
                if circles is not None:
                    for x, y, radius in np.round(circles[0, :]).astype(int):
                        cv2.circle(output, (x, y), radius, (30, 220, 255), 2, cv2.LINE_AA)
                        cv2.drawMarker(output, (x, y), (20, 40, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                return output
            case OperationId.FAST_CORNERS:
                keypoints = cv2.FastFeatureDetector_create(parameter(params, 0, 24), bool(parameter(params, 1, 1))).detect(gray)
                max_points = parameter(params, 2, 500)
                keypoints = sorted(keypoints, key=lambda keypoint: keypoint.response, reverse=True)[:max_points]
                return cv2.drawKeypoints(bgr, keypoints, None, (30, 220, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            case OperationId.ORB_KEYPOINTS:
                detector = cv2.ORB_create(nfeatures=parameter(params, 0, 600), fastThreshold=parameter(params, 1, 20))
                keypoints, _ = detector.detectAndCompute(gray, None)
                return cv2.drawKeypoints(bgr, keypoints, None, (30, 220, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            case OperationId.BACKGROUND_SUBTRACTION:
                return self.background_subtraction(bgr, step, params)
            case OperationId.RUNNING_AVERAGE_FOREGROUND:
                return self.running_average_foreground(bgr, gray, step, params)
            case OperationId.MOTION_BRIGHT_MASK:
                return self.motion_bright_mask(bgr, gray, step, params)
            case OperationId.FRAME_DIFFERENCE:
                return self.frame_difference(bgr, gray, step, params)
            case OperationId.SPARSE_OPTICAL_FLOW:
                return self.sparse_optical_flow(bgr, gray, step, params)
            case OperationId.OPTICAL_FLOW:
                return self.dense_optical_flow(bgr, gray, step, params)
            case OperationId.GOOD_FEATURES:
                corners = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=parameter(params, 0, 120),
                    qualityLevel=parameter(params, 1, 10) / 1000.0,
                    minDistance=parameter(params, 2, 12),
                )
                output = bgr.copy()
                if corners is not None:
                    for corner in corners.reshape(-1, 2):
                        center = (int(round(corner[0])), int(round(corner[1])))
                        cv2.circle(output, center, 4, (30, 220, 255), 2, cv2.LINE_AA)
                        cv2.circle(output, center, 1, (20, 30, 40), -1, cv2.LINE_AA)
                return output

        return frame.copy()

    def best_bright_candidate(self, bgr: np.ndarray, gray: np.ndarray, params: list[int]) -> np.ndarray:
        mask = clean_mask(binary_mask(gray, parameter(params, 0, 220)), parameter(params, 4, 3), parameter(params, 5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(parameter(params, 1, 120))
        max_area = float(parameter(params, 2, 0))
        min_circularity = parameter(params, 3, 30) / 100.0
        best_contour = None
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or (max_area > 0.0 and area > max_area):
                continue
            circularity = contour_circularity(contour)
            if circularity < min_circularity:
                continue
            region_mask = np.zeros_like(mask)
            cv2.drawContours(region_mask, [contour], -1, 255, cv2.FILLED)
            mean_brightness = cv2.mean(gray, region_mask)[0]
            score = mean_brightness * math.sqrt(max(area, 1.0)) * (0.5 + circularity)
            if score > best_score:
                best_score = score
                best_contour = contour

        output = overlay_mask(bgr, mask, (50, 150, 255), 0.18)
        if best_contour is not None:
            x, y, width, height = cv2.boundingRect(best_contour)
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 255), 3)
            cv2.drawContours(output, [best_contour], -1, (30, 230, 255), 2)
            moments = cv2.moments(best_contour)
            if abs(moments["m00"]) > sys.float_info.epsilon:
                center = (int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"])))
                cv2.drawMarker(output, center, (20, 40, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        return output

    def background_subtraction(self, bgr: np.ndarray, step: PipelineStep, params: list[int]) -> np.ndarray:
        history = parameter(params, 0, 120)
        threshold = parameter(params, 1, 16)
        min_area = parameter(params, 2, 600)

        if step.background_subtractor is None or step.background_history != history or step.background_threshold != threshold:
            step.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=float(threshold), detectShadows=True)
            step.background_history = history
            step.background_threshold = threshold

        foreground = step.background_subtractor.apply(bgr)
        _, foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
        foreground = clean_mask(foreground, 5, 0, 2)
        output = overlay_mask(bgr, foreground, (25, 25, 180), 0.32)
        draw_filtered_contours(output, foreground, min_area, 0.0, 0.0, (40, 220, 255))
        return output

    def running_average_foreground(self, bgr: np.ndarray, gray: np.ndarray, step: PipelineStep, params: list[int]) -> np.ndarray:
        gray_float = gray.astype(np.float32)
        output = bgr.copy()
        if step.running_average is None:
            step.running_average = gray_float.copy()
            draw_waiting_label(output, "Need next frame")
            return output

        background = np.clip(step.running_average, 0, 255).astype(np.uint8)
        delta = cv2.absdiff(gray, background)
        mask = clean_mask(binary_mask(delta, parameter(params, 1, 28)), 3, 5, parameter(params, 3, 2))
        output = overlay_mask(bgr, mask, (40, 190, 255), 0.32)
        draw_filtered_contours(output, mask, parameter(params, 2, 400), 0.0, 0.0, (30, 220, 255))
        cv2.accumulateWeighted(gray_float, step.running_average, parameter(params, 0, 40) / 1000.0)
        return output

    def motion_bright_mask(self, bgr: np.ndarray, gray: np.ndarray, step: PipelineStep, params: list[int]) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        output = bgr.copy()
        if step.previous_gray is None:
            step.previous_gray = blurred.copy()
            draw_waiting_label(output, "Need next frame")
            return output

        motion = cv2.absdiff(step.previous_gray, blurred)
        motion = binary_mask(motion, parameter(params, 0, 25))
        bright = binary_mask(gray, parameter(params, 1, 210))
        mask = cv2.bitwise_and(motion, bright)
        mask = clean_mask(mask, parameter(params, 4, 3), parameter(params, 5, 5), parameter(params, 3, 2))
        output = overlay_mask(bgr, mask, (40, 190, 255), 0.35)
        draw_filtered_contours(output, mask, parameter(params, 2, 250), 0.0, 0.0, (30, 220, 255))
        step.previous_gray = blurred.copy()
        return output

    def frame_difference(self, bgr: np.ndarray, gray: np.ndarray, step: PipelineStep, params: list[int]) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        output = bgr.copy()
        if step.previous_gray is None:
            step.previous_gray = blurred.copy()
            draw_waiting_label(output, "Need next frame")
            return output

        delta = cv2.absdiff(step.previous_gray, blurred)
        delta = binary_mask(delta, parameter(params, 0, 32))
        dilation = parameter(params, 2, 2)
        if dilation > 0:
            delta = cv2.dilate(delta, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=dilation)
        output = overlay_mask(bgr, delta, (30, 30, 210), 0.28)
        draw_filtered_contours(output, delta, parameter(params, 1, 450), 0.0, 0.0, (40, 230, 255))
        step.previous_gray = blurred.copy()
        return output

    def sparse_optical_flow(self, bgr: np.ndarray, gray: np.ndarray, step: PipelineStep, params: list[int]) -> np.ndarray:
        output = bgr.copy()
        max_points = parameter(params, 0, 200)
        quality = parameter(params, 1, 10) / 1000.0
        min_distance = parameter(params, 2, 10)
        min_move = parameter(params, 3, 1)

        if step.previous_gray is None or step.previous_points is None or len(step.previous_points) == 0:
            step.previous_points = cv2.goodFeaturesToTrack(gray, max_points, quality, min_distance)
            step.previous_gray = gray.copy()
            if step.previous_points is not None:
                for point in step.previous_points.reshape(-1, 2):
                    cv2.circle(output, (int(round(point[0])), int(round(point[1]))), 3, (30, 220, 255), -1, cv2.LINE_AA)
            draw_waiting_label(output, "Need next frame")
            return output

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(step.previous_gray, gray, step.previous_points, None)
        good_points: list[list[float]] = []
        if next_points is not None and status is not None:
            previous_flat = step.previous_points.reshape(-1, 2)
            next_flat = next_points.reshape(-1, 2)
            status_flat = status.reshape(-1)

            for previous, current, ok in zip(previous_flat, next_flat, status_flat, strict=False):
                if not ok:
                    continue
                if current[0] < 0 or current[1] < 0 or current[0] >= gray.shape[1] or current[1] >= gray.shape[0]:
                    continue
                movement = math.hypot(float(current[0] - previous[0]), float(current[1] - previous[1]))
                if movement >= min_move:
                    cv2.arrowedLine(
                        output,
                        (int(round(previous[0])), int(round(previous[1]))),
                        (int(round(current[0])), int(round(current[1]))),
                        (40, 220, 255),
                        1,
                        cv2.LINE_AA,
                        tipLength=0.25,
                    )
                cv2.circle(output, (int(round(current[0])), int(round(current[1]))), 3, (20, 40, 255), -1, cv2.LINE_AA)
                good_points.append([float(current[0]), float(current[1])])

        if len(good_points) < max(10, max_points // 4):
            refreshed = cv2.goodFeaturesToTrack(gray, max_points, quality, min_distance)
            step.previous_points = refreshed
        else:
            step.previous_points = np.array(good_points, dtype=np.float32).reshape(-1, 1, 2)
        step.previous_gray = gray.copy()
        return output

    def dense_optical_flow(self, bgr: np.ndarray, gray: np.ndarray, step: PipelineStep, params: list[int]) -> np.ndarray:
        output = bgr.copy()
        if step.previous_gray is None:
            step.previous_gray = gray.copy()
            draw_waiting_label(output, "Need next frame")
            return output

        flow = cv2.calcOpticalFlowFarneback(step.previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        grid = parameter(params, 0, 24)
        gain = parameter(params, 1, 18) / 10.0
        min_magnitude = parameter(params, 2, 3)
        for y in range(grid // 2, output.shape[0], grid):
            for x in range(grid // 2, output.shape[1], grid):
                vector_x, vector_y = flow[y, x]
                magnitude = math.hypot(float(vector_x), float(vector_y))
                if magnitude < min_magnitude:
                    continue
                end = (int(round(x + vector_x * gain)), int(round(y + vector_y * gain)))
                cv2.arrowedLine(output, (x, y), end, (40, 220, 255), 1, cv2.LINE_AA, tipLength=0.32)
        step.previous_gray = gray.copy()
        return output


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("OpenCV Operation Viewer - PyQt")
    app.setOrganizationName("opencv-viewer")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
