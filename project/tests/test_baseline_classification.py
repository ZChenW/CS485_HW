import unittest

import pandas as pd

from src.baseline_classification import (
    evaluate_predictions,
    make_stratified_splits,
    train_majority_baseline,
)


class BaselineClassificationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "text": [f"text {label} {idx}" for label in ["a", "b", "c"] for idx in range(20)],
                "label": [label for label in ["a", "b", "c"] for _ in range(20)],
            }
        )

    def test_stratified_splits_are_reproducible_and_cover_all_rows(self):
        first = make_stratified_splits(self.df, seed=123)
        second = make_stratified_splits(self.df, seed=123)

        self.assertEqual([len(split) for split in first], [42, 9, 9])
        self.assertEqual(
            [split["text"].tolist() for split in first],
            [split["text"].tolist() for split in second],
        )
        for split in first:
            self.assertEqual(set(split["label"]), {"a", "b", "c"})

    def test_majority_baseline_predicts_training_majority_label(self):
        df = pd.DataFrame(
            {
                "text": ["one", "two", "three", "four"],
                "label": ["majority", "majority", "minority", "majority"],
            }
        )

        model = train_majority_baseline(df)
        predictions = model.predict(["new text", "another"])

        self.assertEqual(predictions.tolist(), ["majority", "majority"])

    def test_evaluate_predictions_returns_metrics_and_misclassified_rows(self):
        split = pd.DataFrame(
            {
                "text": ["first", "second", "third"],
                "label": ["a", "b", "b"],
            }
        )

        metrics, report, misclassified = evaluate_predictions(
            split,
            predictions=["a", "a", "b"],
            labels=["a", "b"],
        )

        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("a", report)
        self.assertEqual(misclassified.to_dict("records"), [{"text": "second", "label": "b", "prediction": "a"}])


if __name__ == "__main__":
    unittest.main()
