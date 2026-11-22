"""Unit tests for Set 9 (YOLO): label parsing, drawing, counting and the tools end to end."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools import yolo_detection as Y

NAMES = {0: "PC", 1: "RBC", 2: "WBC"}
HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


def _write_split(root: Path, n_images: int = 3) -> Path:
    """A tiny YOLO-layout split: images/ with coloured squares and labels/ describing them."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for k in range(n_images):
        image = Image.new("RGB", (100, 80), (240, 220, 210))
        image.paste((200, 60, 60), (10, 10, 30, 30))  # "RBC"
        image.paste((90, 50, 160), (50, 20, 90, 70))  # "WBC"
        image.save(root / "images" / f"img{k}.png")
        (root / "labels" / f"img{k}.txt").write_text("1 0.2 0.25 0.2 0.25\n2 0.7 0.5625 0.4 0.625\n", encoding="utf-8")
    return root / "images"


class Set9RegistryTests(unittest.TestCase):
    def test_set9_tools_are_registered_and_grouped(self) -> None:
        grouped = dict(build_tool_registry().categories())
        ids = [t.tool_id for t in grouped[Y.CATEGORY]]
        self.assertEqual(
            ids,
            ["yolo_pretrained", "yolo_blood_cells", "yolo_cell_count", "yolo_activation_map"],
        )
        for tool in grouped[Y.CATEGORY]:
            self.assertFalse(tool.requires_image)


class Set9HelperTests(unittest.TestCase):
    def test_label_path_and_yolo_labels_to_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = _write_split(Path(tmp))
            image_path = images / "img0.png"
            self.assertEqual(Y.label_path_for(image_path), Path(tmp) / "labels" / "img0.txt")
            gt = Y.read_yolo_labels(Y.label_path_for(image_path), 100, 80, NAMES)
            np.testing.assert_allclose(gt.boxes[0], [10, 10, 30, 30])
            np.testing.assert_allclose(gt.boxes[1], [50, 20, 90, 70])
            self.assertEqual(gt.counts(), {"RBC": 1, "WBC": 1})
            self.assertIsNone(Y.ground_truth_for(Path(tmp) / "nowhere.png", Image.new("RGB", (4, 4)), NAMES))

    def test_detections_filter_count_table_and_drawing(self) -> None:
        pred = Y.Detections(
            boxes=np.array([[1, 1, 20, 20], [30, 30, 60, 60], [5, 40, 15, 50]], float),
            classes=np.array([1, 1, 0]),
            names=NAMES,
            scores=np.array([0.9, 0.2, 0.6]),
        )
        self.assertEqual(len(pred.above(0.5)), 2)
        gt = Y.Detections(np.array([[1, 1, 20, 20]], float), np.array([1]), NAMES)
        table = Y.count_table(NAMES, pred, gt)
        self.assertEqual(table, {"PC": (1, 0), "RBC": (2, 1), "WBC": (0, 0)})
        image = Image.new("RGB", (80, 80), "white")
        drawn = Y.gt_vs_pred_image(image, gt, pred)
        self.assertEqual(drawn.size, image.size)
        self.assertNotEqual(drawn.tobytes(), image.tobytes())
        fig = Y.count_figure(drawn, table, "test")
        self.assertEqual(len(fig.axes), 2)

    def test_letterbox_crop_removes_padding(self) -> None:
        cam = np.zeros((64, 64))
        cam[16:48, :] = 1.0  # a 2:1 image letterboxed into a square: padding above and below
        heat = Y.letterbox_crop(cam, 200, 100)
        self.assertEqual(heat.shape, (100, 200))
        self.assertGreater(heat.min(), 0.9)

    def test_count_scatter_figure(self) -> None:
        fig = Y.count_scatter_figure({"PC": [0, 1], "RBC": [10, 12]}, {"PC": [0, 2], "RBC": [11, 12]}, 0.25)
        self.assertEqual(len(fig.axes), 2)


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray) -> None:
        import torch

        self.xyxy, self.cls, self.conf = torch.tensor(xyxy), torch.tensor(cls), torch.tensor(conf)


class _FakeResult:
    def __init__(self) -> None:
        self.boxes = _FakeBoxes(np.array([[10.0, 10, 30, 30]]), np.array([1.0]), np.array([0.8]))


class _FakeYolo:
    names = NAMES

    def predict(self, *_args: object, **_kwargs: object) -> list[_FakeResult]:
        return [_FakeResult()]


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch not installed")
class Set9ToolTests(unittest.TestCase):
    def _run(self, tool: Y.ForensicsTool, settings: Y.YoloSettings, doc: ImageDocument) -> Y.ToolResult:
        with (
            patch.object(Y, "YoloDialog") as dialog,
            patch.object(Y, "load_yolo_model", return_value=_FakeYolo()),
        ):
            dialog.return_value.result = settings
            result = tool.run(None, doc)
        assert result is not None
        return result

    def test_blood_cell_and_count_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = _write_split(Path(tmp))
            doc = ImageDocument()
            doc.load(images / "img0.png")

            result = self._run(Y.YoloBloodCellTool(), Y.YoloSettings(model_path="fake.pt"), doc)
            self.assertIn("RBC 1/1", result.message)
            self.assertEqual(result.image.size, (100, 80))

            result = self._run(Y.CellCountTool(), Y.YoloSettings(model_path="fake.pt"), doc)
            self.assertEqual(result.details["  WBC"], "0 predicted / 1 GT")

            folder = Y.YoloSettings(model_path="fake.pt", source="folder", folder=str(images), max_images=10)
            result = self._run(Y.CellCountTool(), folder, doc)
            self.assertIn("3 with labels", result.details["Images"])

    def test_pretrained_tool_on_image_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = _write_split(Path(tmp))
            settings = Y.YoloSettings(model_path="yolo11n.pt", source="file", image_path=str(images / "img1.png"))
            result = self._run(Y.YoloPretrainedTool(), settings, ImageDocument())
            self.assertEqual(result.details["Detections"], 1)
            self.assertIn("1 RBC", result.message)


@unittest.skipUnless(HAS_ULTRALYTICS and Y.default_custom_model(), "ultralytics or trained blood-cell model missing")
class Set9RealModelTests(unittest.TestCase):
    def test_activation_map_with_trained_model(self) -> None:
        model = Y.load_yolo_model(Y.default_custom_model())
        image = Image.new("RGB", (320, 240), (230, 200, 200))
        detections, cam = Y.activation_map(model, image, 16)
        self.assertEqual(cam.ndim, 2)
        self.assertAlmostEqual(float(cam.max()), 1.0, places=3)
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertEqual(len(Y.activation_figure(image, detections, cam, 16, "C3k2").axes), 4)  # 3 panels + colorbar
        with self.assertRaises(ValueError):
            Y.activation_map(model, image, 99)

    def test_activation_tool_compares_detection_scales(self) -> None:
        layers = Y.detection_scale_layers(Y.load_yolo_model(Y.default_custom_model()), Image.new("RGB", (64, 64)))
        self.assertEqual(layers, [16, 19, 22])
        settings = Y.YoloSettings(model_path=Y.default_custom_model(), source="file", compare_layers=True,
                                  image_path=str(sorted(Y.DEFAULT_BLOOD_TEST_DIR.iterdir())[0]))
        with patch.object(Y, "YoloDialog") as dialog:
            dialog.return_value.result = settings
            result = Y.YoloActivationMapTool().run(None, ImageDocument())
        assert result is not None
        self.assertIn("Layer 22 (C3k2)", " ".join(result.details))
        self.assertIn("80 × 80", result.details["  Layer 16 (C3k2)"])


if __name__ == "__main__":
    unittest.main()
