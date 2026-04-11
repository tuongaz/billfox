"""YOLO-based document cropping preprocessor using ONNX Runtime."""

from __future__ import annotations

import io
from typing import Any

from billfox._types import Document

_IMAGE_MIME_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/heic",
})


class YOLOPreprocessor:
    """Crop documents using a YOLO object detection ONNX model.

    Only processes image MIME types; PDFs pass through unchanged.
    If no detections are found, returns the original document.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        self._model_path = model_path
        self._confidence = confidence
        self._imgsz = imgsz
        self._session: Any = None

    def _get_session(self) -> Any:
        """Lazy-load the ONNX Runtime inference session."""
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError:
                raise RuntimeError(
                    "onnxruntime is required for YOLOPreprocessor. "
                    "Install it with: pip install billfox[yolo]"
                ) from None

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    async def process(self, document: Document) -> Document:
        """Crop the document image using YOLO detection.

        Non-image documents are returned unchanged.
        If no detections are found, the original document is returned.
        """
        if document.mime_type not in _IMAGE_MIME_TYPES:
            return document

        import numpy as np
        from PIL import Image

        # Decode image bytes to numpy BGR array
        pil_img = Image.open(io.BytesIO(document.content)).convert("RGB")
        img_array = np.array(pil_img)
        # Convert RGB to BGR for YOLO processing
        bgr = img_array[:, :, ::-1].copy()

        # Run detection
        crops = _crop_detections(
            self._get_session(),
            bgr,
            conf=self._confidence,
            imgsz=self._imgsz,
        )

        if not crops:
            return document

        # Use the first (highest confidence) crop
        crop_bgr = crops[0]
        # Convert BGR back to RGB for PIL
        crop_rgb = crop_bgr[:, :, ::-1]
        crop_pil = Image.fromarray(crop_rgb)
        buf = io.BytesIO()
        crop_pil.save(buf, format="JPEG")
        cropped_bytes = buf.getvalue()

        return Document(
            content=cropped_bytes,
            mime_type="image/jpeg",
            source_uri=document.source_uri,
            metadata={**document.metadata, "preprocessor": "yolo"},
        )


# ---------------------------------------------------------------------------
# Ported from billfox-app/api/src/ai/yolo/receipt_onnx.py
# Renamed from receipt-specific to generic document naming.
# ---------------------------------------------------------------------------

_Box = tuple[int, int, int, int]
_BoxF = tuple[float, float, float, float]


def _preprocess(
    image: Any,
    imgsz: int = 640,
) -> tuple[Any, tuple[int, int], float]:
    """Preprocess image for YOLO ONNX inference.

    Returns (tensor, (orig_h, orig_w), scale).
    """
    import numpy as np

    orig_h, orig_w = image.shape[:2]
    scale = min(imgsz / orig_h, imgsz / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize using PIL to avoid cv2 dependency
    from PIL import Image

    bgr = image
    rgb = bgr[:, :, ::-1]
    pil_img = Image.fromarray(rgb)
    pil_resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_rgb = np.array(pil_resized)

    # Letterbox padding
    padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_rgb

    # Normalise to [0, 1] and transpose to (C, H, W)
    tensor = padded.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, (orig_h, orig_w), scale


def _compute_iou(box1: _BoxF, box2: _BoxF) -> float:
    """Compute Intersection over Union between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _compute_containment(box1: _BoxF, box2: _BoxF) -> float:
    """Compute fraction of box2 contained within box1."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    return inter_area / area2 if area2 > 0 else 0.0


def _nms(
    boxes: list[_Box],
    confidences: list[float],
    iou_threshold: float = 0.45,
    containment_threshold: float = 0.8,
) -> list[_Box]:
    """Apply Non-Maximum Suppression using IoU and containment."""
    if not boxes:
        return []

    indices = sorted(range(len(boxes)), key=lambda i: confidences[i], reverse=True)
    keep: list[_Box] = []

    while indices:
        current = indices.pop(0)
        keep.append(boxes[current])

        remaining: list[int] = []
        for idx in indices:
            current_box: _BoxF = (
                float(boxes[current][0]),
                float(boxes[current][1]),
                float(boxes[current][2]),
                float(boxes[current][3]),
            )
            other_box: _BoxF = (
                float(boxes[idx][0]),
                float(boxes[idx][1]),
                float(boxes[idx][2]),
                float(boxes[idx][3]),
            )
            iou = _compute_iou(current_box, other_box)
            containment = _compute_containment(current_box, other_box)
            if iou < iou_threshold and containment < containment_threshold:
                remaining.append(idx)
        indices = remaining

    return keep


def _postprocess(
    output: Any,
    orig_shape: tuple[int, int],
    scale: float,
    imgsz: int = 640,
    conf: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[_Box]:
    """Postprocess YOLO output to bounding boxes in original image coordinates."""
    orig_h, orig_w = orig_shape
    pad_w = (imgsz - int(orig_w * scale)) // 2
    pad_h = (imgsz - int(orig_h * scale)) // 2

    predictions = output[0].T  # (num_boxes, 5)

    boxes: list[_Box] = []
    confidences: list[float] = []

    for pred in predictions:
        x_center, y_center, width, height, confidence = pred
        if confidence < conf:
            continue

        x1 = x_center - width / 2 - pad_w
        y1 = y_center - height / 2 - pad_h
        x2 = x_center + width / 2 - pad_w
        y2 = y_center + height / 2 - pad_h

        x1_i = max(0, min(int(x1 / scale), orig_w))
        y1_i = max(0, min(int(y1 / scale), orig_h))
        x2_i = max(0, min(int(x2 / scale), orig_w))
        y2_i = max(0, min(int(y2 / scale), orig_h))

        if x2_i > x1_i and y2_i > y1_i:
            boxes.append((x1_i, y1_i, x2_i, y2_i))
            confidences.append(float(confidence))

    return _nms(boxes, confidences, iou_threshold)


def _crop_detections(
    session: Any,
    image: Any,
    conf: float = 0.25,
    imgsz: int = 640,
) -> list[Any]:
    """Run YOLO ONNX inference and return cropped detections as BGR arrays."""
    tensor, orig_shape, scale = _preprocess(image, imgsz)

    input_name: str = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})

    boxes = _postprocess(outputs[0], orig_shape, scale, imgsz, conf)

    crops = []
    for x1, y1, x2, y2 in boxes:
        crop = image[y1:y2, x1:x2].copy()
        crops.append(crop)
    return crops
