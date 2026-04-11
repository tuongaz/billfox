"""Tests for YOLOPreprocessor with mocked ONNX session."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from billfox._types import Document
from billfox.preprocess.yolo import (
    YOLOPreprocessor,
    _compute_containment,
    _compute_iou,
    _nms,
    _postprocess,
    _preprocess,
)


def _make_image_bytes(width: int = 100, height: int = 80, color: str = "red") -> bytes:
    """Create a test JPEG image as bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_document(
    mime_type: str = "image/jpeg",
    width: int = 100,
    height: int = 80,
) -> Document:
    """Create a test Document with image content."""
    return Document(
        content=_make_image_bytes(width, height),
        mime_type=mime_type,
        source_uri="/test/image.jpg",
    )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestComputeIoU:
    def test_identical_boxes(self) -> None:
        box = (0.0, 0.0, 10.0, 10.0)
        assert _compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert _compute_iou((0.0, 0.0, 5.0, 5.0), (10.0, 10.0, 20.0, 20.0)) == 0.0

    def test_partial_overlap(self) -> None:
        iou = _compute_iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0))
        # Intersection = 5*5=25, Union = 100+100-25=175
        assert iou == pytest.approx(25.0 / 175.0)


class TestComputeContainment:
    def test_fully_contained(self) -> None:
        outer = (0.0, 0.0, 100.0, 100.0)
        inner = (10.0, 10.0, 50.0, 50.0)
        assert _compute_containment(outer, inner) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert _compute_containment((0.0, 0.0, 5.0, 5.0), (10.0, 10.0, 20.0, 20.0)) == 0.0


class TestNMS:
    def test_empty_input(self) -> None:
        assert _nms([], []) == []

    def test_single_box(self) -> None:
        boxes = [(0, 0, 10, 10)]
        assert _nms(boxes, [0.9]) == boxes

    def test_suppresses_overlapping(self) -> None:
        boxes = [(0, 0, 10, 10), (1, 1, 11, 11)]
        result = _nms(boxes, [0.9, 0.5], iou_threshold=0.3)
        assert len(result) == 1
        assert result[0] == (0, 0, 10, 10)  # higher confidence kept

    def test_keeps_non_overlapping(self) -> None:
        boxes = [(0, 0, 10, 10), (50, 50, 60, 60)]
        result = _nms(boxes, [0.9, 0.8])
        assert len(result) == 2


class TestPreprocess:
    def test_output_shape(self) -> None:
        img = np.zeros((100, 80, 3), dtype=np.uint8)
        tensor, orig_shape, scale = _preprocess(img, imgsz=640)
        assert tensor.shape == (1, 3, 640, 640)
        assert orig_shape == (100, 80)
        assert scale > 0

    def test_preserves_original_dimensions(self) -> None:
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        _, orig_shape, _ = _preprocess(img, imgsz=640)
        assert orig_shape == (200, 300)


class TestPostprocess:
    def test_filters_low_confidence(self) -> None:
        # Single detection with low confidence
        output = np.zeros((1, 5, 1), dtype=np.float32)
        output[0, :, 0] = [320, 320, 100, 100, 0.1]  # conf=0.1 < 0.25
        boxes = _postprocess(output, (640, 640), 1.0, 640, conf=0.25)
        assert boxes == []

    def test_returns_valid_boxes(self) -> None:
        # Single detection with high confidence, centered in image
        output = np.zeros((1, 5, 1), dtype=np.float32)
        output[0, :, 0] = [320, 320, 100, 100, 0.9]
        boxes = _postprocess(output, (640, 640), 1.0, 640, conf=0.25)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert x2 > x1
        assert y2 > y1


# ---------------------------------------------------------------------------
# Integration-level tests for YOLOPreprocessor
# ---------------------------------------------------------------------------


class TestYOLOPreprocessor:
    def _mock_session(self, detections: list[tuple[float, float, float, float, float]]) -> MagicMock:
        """Create a mock ONNX session that returns specified detections.

        Each detection is (x_center, y_center, width, height, confidence).
        """
        output = np.zeros((1, 5, len(detections)), dtype=np.float32)
        for i, det in enumerate(detections):
            output[0, :, i] = det

        mock_input = MagicMock()
        mock_input.name = "images"

        session = MagicMock()
        session.get_inputs.return_value = [mock_input]
        session.run.return_value = [output]
        return session

    @pytest.mark.asyncio
    async def test_passes_pdf_through(self) -> None:
        doc = Document(
            content=b"%PDF-1.4 ...",
            mime_type="application/pdf",
            source_uri="/test/doc.pdf",
        )
        preprocessor = YOLOPreprocessor(model_path="/fake/model.onnx")
        result = await preprocessor.process(doc)
        assert result is doc  # exact same object

    @pytest.mark.asyncio
    async def test_returns_original_when_no_detections(self) -> None:
        doc = _make_document()
        preprocessor = YOLOPreprocessor(model_path="/fake/model.onnx")
        preprocessor._session = self._mock_session([])
        result = await preprocessor.process(doc)
        assert result is doc

    @pytest.mark.asyncio
    async def test_crops_image_on_detection(self) -> None:
        width, height = 640, 640
        doc = _make_document(width=width, height=height)
        # Detection centered at (320, 320), size 200x200, high confidence
        preprocessor = YOLOPreprocessor(model_path="/fake/model.onnx")
        preprocessor._session = self._mock_session([(320, 320, 200, 200, 0.9)])

        result = await preprocessor.process(doc)

        assert result is not doc
        assert result.mime_type == "image/jpeg"
        assert result.source_uri == doc.source_uri
        assert result.metadata.get("preprocessor") == "yolo"
        # Verify it's valid JPEG
        img = Image.open(io.BytesIO(result.content))
        assert img.format == "JPEG"

    @pytest.mark.asyncio
    async def test_configurable_confidence(self) -> None:
        doc = _make_document(width=640, height=640)
        # Detection with confidence 0.3 — below high threshold
        preprocessor = YOLOPreprocessor(
            model_path="/fake/model.onnx",
            confidence=0.5,
        )
        preprocessor._session = self._mock_session([(320, 320, 200, 200, 0.3)])

        result = await preprocessor.process(doc)
        # Should return original since 0.3 < 0.5 confidence threshold
        assert result is doc

    @pytest.mark.asyncio
    async def test_lazy_import_error(self) -> None:
        preprocessor = YOLOPreprocessor(model_path="/fake/model.onnx")
        with (
            patch.dict("sys.modules", {"onnxruntime": None}),
            pytest.raises(RuntimeError, match="onnxruntime is required"),
        ):
            preprocessor._get_session()

    @pytest.mark.asyncio
    async def test_protocol_conformance(self) -> None:
        from billfox.preprocess import Preprocessor

        preprocessor = YOLOPreprocessor(model_path="/fake/model.onnx")
        assert isinstance(preprocessor, Preprocessor)
