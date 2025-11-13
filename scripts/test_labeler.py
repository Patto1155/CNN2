"""Unit tests for the deterministic labeler."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts import label_windows


DATASET_PATH = Path("data/dataset_labeled.npz")


class LabelerDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"{DATASET_PATH} is missing. Run scripts/label_windows.py before executing tests."
            )
        data = np.load(DATASET_PATH)
        cls.X = data["X"]
        cls.y = data["y"]

    def test_dataset_shapes(self) -> None:
        self.assertEqual(self.X.shape[0], self.y.shape[0], "Sample counts must match.")
        self.assertEqual(self.y.shape[1], 4, "Label tensor must have 4 heads.")

    def test_labels_are_binary(self) -> None:
        unique_values = np.unique(self.y)
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])), f"Labels contain non-binary values {unique_values}.")

    def test_each_head_has_positives(self) -> None:
        positives = self.y.sum(axis=0)
        for idx, name in enumerate(label_windows.LABEL_NAMES):
            self.assertGreater(positives[idx], 0, f"{name} head has no positives.")

    def test_continuation_references_retest(self) -> None:
        cont_indices = np.where(self.y[:, 3] == 1)[0]
        self.assertGreater(cont_indices.size, 0, "No continuation samples available for dependency test.")
        sample_indices = cont_indices[: min(10, cont_indices.size)]
        windows = self.X[sample_indices]
        params = label_windows.load_labeler_params()
        labels, diagnostics = label_windows.label_dataset(
            windows, params=params, return_diagnostics=True
        )
        self.assertIsNotNone(diagnostics, "Diagnostics must be returned when requested.")
        assert diagnostics is not None  # mypy hint
        for diag, label_vec in zip(diagnostics, labels):
            self.assertEqual(label_vec[3], 1, "Sample should be labeled as continuation.")
            self.assertIsNotNone(diag.t_retest, "Continuation sample missing retest index.")
            self.assertIsNotNone(diag.t_continuation, "Continuation sample missing continuation index.")
            self.assertGreaterEqual(
                diag.t_continuation,
                diag.t_retest,
                "Continuation must occur after the retest reclaim.",
            )


if __name__ == "__main__":
    unittest.main()
