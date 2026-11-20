"""Unit tests for Set 8 tools: Pretrained CNN & Model Passport (8.2) and Feature Projection (8.3)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.cnn_classification import (
    DEFAULT_SET8_DIR,
    ModelPassport,
    PretrainedCnnSettings,
    PretrainedCnnTool,
    classification_summary_figure,
    classify_image_with_model,
    fallback_passport_for_model,
    find_passport_for_model,
    load_model_passport,
    passport_history_figure,
    preprocess_for_passport,
)
from forensics_app.tools.feature_projection import (
    FeatureProjectionSettings,
    FeatureProjectionTool,
    load_feature_vectors,
    project_tsne,
    projection_figure,
    subsample_stratified,
)


class _DummyKerasModel:
    """Fast stand-in for a Keras model in unit tests."""

    input_shape = (None, 32, 32, 3)
    output_shape = (None, 3)

    def __init__(self, probs: list[float]) -> None:
        self._probs = np.array([probs], dtype=np.float32)
        self.last_batch_shape: tuple[int, ...] | None = None

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.last_batch_shape = tuple(batch.shape)
        return self._probs


class Set8RegistryTests(unittest.TestCase):
    def test_set8_tools_are_registered_and_grouped(self) -> None:
        registry = build_tool_registry()
        grouped = dict(registry.categories())
        self.assertIn("Set 8: Deep Learning & Feature Projection", grouped)
        ids = [t.tool_id for t in grouped["Set 8: Deep Learning & Feature Projection"]]
        self.assertEqual(ids, ["pretrained_cnn", "feature_projection"])
        for tool in grouped["Set 8: Deep Learning & Feature Projection"]:
            self.assertFalse(tool.requires_image)


class Set8CnnTests(unittest.TestCase):
    def _sample_passport_dict(self) -> dict:
        return {
            "model_name": "Test CNN",
            "architecture": "Conv -> Dense",
            "model_file": "test_cnn.keras",
            "dataset": "CIFAR-10",
            "sample_size": {"train": 900, "validation": 100, "test": 200},
            "class_names": ["cat", "dog", "bird"],
            "preprocessing": {
                "input_height": 32,
                "input_width": 32,
                "channels": 3,
                "color_mode": "RGB",
                "normalization": "rescale_0_1",
                "description": "Resize 32x32 and divide by 255.",
            },
            "training_history": {
                "loss": [1.1, 0.7, 0.5],
                "val_loss": [1.0, 0.8, 0.6],
                "accuracy": [0.4, 0.65, 0.8],
                "val_accuracy": [0.45, 0.6, 0.75],
            },
            "test_metrics": {"test_loss": 0.62, "test_accuracy": 0.74},
        }

    def test_passport_load_and_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "my_model.keras"
            model_path.write_bytes(b"fake")
            passport_path = root / "my_model_passport.json"
            passport_path.write_text(json.dumps(self._sample_passport_dict()), encoding="utf-8")

            found = find_passport_for_model(model_path)
            self.assertEqual(found, passport_path)

            passport = load_model_passport(passport_path)
            self.assertEqual(passport.model_name, "Test CNN")
            self.assertEqual(passport.class_names, ("cat", "dog", "bird"))
            self.assertEqual(passport.input_height, 32)
            self.assertEqual(passport.normalization, "rescale_0_1")
            keys = [k for k, _ in passport.summary_lines()]
            self.assertIn("Dataset", keys)
            self.assertIn("Sample size", keys)
            self.assertIn("Preprocessing", keys)
            self.assertIn("Training history", keys)

            fig = passport_history_figure(passport)
            self.assertEqual(len(fig.axes), 2)

    def test_preprocess_rescale_and_inception_minus1_1(self) -> None:
        img = Image.new("RGB", (64, 48), (255, 0, 128))
        p_01 = ModelPassport(
            model_name="cnn",
            preprocessing={"input_height": 32, "input_width": 32, "channels": 3, "normalization": "rescale_0_1"},
        )
        batch_01 = preprocess_for_passport(img, p_01)
        self.assertEqual(batch_01.shape, (1, 32, 32, 3))
        self.assertAlmostEqual(float(batch_01[0, 0, 0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(batch_01[0, 0, 0, 1]), 0.0, places=4)

        p_inc = ModelPassport(
            model_name="inc",
            preprocessing={
                "input_height": 128,
                "input_width": 128,
                "channels": 3,
                "normalization": "inception_v3_minus1_1",
            },
        )
        batch_inc = preprocess_for_passport(img, p_inc)
        self.assertEqual(batch_inc.shape, (1, 128, 128, 3))
        self.assertAlmostEqual(float(batch_inc[0, 0, 0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(batch_inc[0, 0, 0, 1]), -1.0, places=4)

    def test_classify_image_and_summary_figure(self) -> None:
        passport = ModelPassport(
            model_name="Test CNN",
            class_names=("cat", "dog", "bird"),
            preprocessing={"input_height": 32, "input_width": 32, "channels": 3, "normalization": "rescale_0_1"},
        )
        dummy = _DummyKerasModel([0.1, 0.75, 0.15])
        img = Image.new("RGB", (50, 50), (100, 150, 200))

        pred, conf, probs = classify_image_with_model(img, dummy, passport)
        self.assertEqual(dummy.last_batch_shape, (1, 32, 32, 3))
        self.assertEqual(pred, "dog")
        self.assertAlmostEqual(conf, 0.75, places=5)
        self.assertEqual(set(probs.keys()), {"cat", "dog", "bird"})

        fig = classification_summary_figure(img, pred, conf, probs, passport)
        self.assertEqual(len(fig.axes), 2)

    def test_fallback_passport_uses_model_shapes(self) -> None:
        dummy = _DummyKerasModel([0.2, 0.3, 0.5])
        passport = fallback_passport_for_model(Path("custom.keras"), dummy)
        self.assertEqual(passport.input_height, 32)
        self.assertEqual(passport.class_names, ("class_0", "class_1", "class_2"))

    def test_pretrained_cnn_tool_run_with_mocked_dialog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "demo.keras"
            model_path.write_bytes(b"fake")
            passport_path = root / "demo_passport.json"
            passport_path.write_text(json.dumps(self._sample_passport_dict()), encoding="utf-8")

            doc = ImageDocument()
            doc.current = Image.new("RGB", (40, 40), "blue")
            settings = PretrainedCnnSettings(
                model_path=str(model_path),
                passport_path=str(passport_path),
                mode="classify",
            )
            dummy = _DummyKerasModel([0.85, 0.10, 0.05])
            with (
                patch("forensics_app.tools.cnn_classification.PretrainedCnnDialog") as mock_dlg,
                patch("forensics_app.tools.cnn_classification.load_keras_model", return_value=dummy),
            ):
                mock_dlg.return_value.result = settings
                result = PretrainedCnnTool().run(None, doc)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertIsNotNone(result.image)
            self.assertEqual(result.details["Predicted class"], "cat")
            self.assertIn("cat", result.message)


class Set8FeatureProjectionTests(unittest.TestCase):
    def _write_synthetic_features(self, path: Path, n_per_class: int = 12, dims: int = 16) -> None:
        rng = np.random.default_rng(0)
        feats, labels = [], []
        for cls_id in range(3):
            center = np.zeros(dims, dtype=np.float32)
            center[cls_id] = 5.0
            feats.append(center + rng.normal(scale=0.3, size=(n_per_class, dims)).astype(np.float32))
            labels.append(np.full(n_per_class, cls_id, dtype=np.int32))
        np.savez_compressed(
            path,
            features=np.concatenate(feats, axis=0),
            labels=np.concatenate(labels, axis=0),
            class_names=np.array(["alpha", "beta", "gamma"]),
            model_name=np.array("SyntheticNet"),
            layer_name=np.array("penultimate"),
            dataset=np.array("Synthetic test"),
        )

    def test_load_subsample_tsne_and_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "features.npz"
            self._write_synthetic_features(npz_path, n_per_class=10, dims=12)

            ds = load_feature_vectors(npz_path)
            self.assertEqual(ds.n_samples, 30)
            self.assertEqual(ds.n_dims, 12)
            self.assertEqual(ds.class_names, ("alpha", "beta", "gamma"))

            sub = subsample_stratified(ds, max_samples=15)
            self.assertEqual(sub.n_samples, 15)

            emb = project_tsne(sub.features, perplexity=5.0, max_iter=250)
            self.assertEqual(emb.shape, (15, 2))
            self.assertTrue(np.all(np.isfinite(emb)))

            fig = projection_figure({"t-SNE": emb}, sub.labels, sub.class_names, subtitle="Test")
            self.assertEqual(len(fig.axes), 1)

    def test_feature_projection_tool_run_tsne(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "features.npz"
            self._write_synthetic_features(npz_path, n_per_class=8, dims=10)

            settings = FeatureProjectionSettings(
                feature_path=str(npz_path),
                method="tsne",
                tsne_perplexity=5.0,
                tsne_max_iter=250,
            )
            with patch("forensics_app.tools.feature_projection.FeatureProjectionDialog") as mock_dlg:
                mock_dlg.return_value.result = settings
                result = FeatureProjectionTool().run(None, ImageDocument())

            self.assertIsNotNone(result)
            assert result is not None
            self.assertIsNotNone(result.image)
            self.assertEqual(result.details["Model"], "SyntheticNet")
            self.assertEqual(result.details["Methods"], "t-SNE")

    def test_exported_set8_artifacts_if_present(self) -> None:
        if not DEFAULT_SET8_DIR.is_dir():
            self.skipTest("models/set8 not generated yet")
        for stem in ("small_cnn_cifar10", "inceptionv3_cifar10"):
            passport_file = DEFAULT_SET8_DIR / f"{stem}_passport.json"
            if passport_file.is_file():
                passport = load_model_passport(passport_file)
                self.assertEqual(len(passport.class_names), 10)
                self.assertTrue(passport.training_history)
        for fname in ("small_cnn_test_features.npz", "inceptionv3_test_features.npz"):
            feat_file = DEFAULT_SET8_DIR / fname
            if feat_file.is_file():
                ds = load_feature_vectors(feat_file)
                self.assertGreaterEqual(ds.n_samples, 100)
                self.assertEqual(len(ds.class_names), 10)


if __name__ == "__main__":
    unittest.main()
