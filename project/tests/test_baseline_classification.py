import unittest

import pandas as pd

from src.baseline_classification import (
    evaluate_predictions,
    make_stratified_splits,
    train_majority_baseline,
)
from src.tune_tfidf_logreg import (
    best_candidate_from_dev_results,
    build_tfidf_logreg_pipeline,
    summarize_misclassification_patterns,
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

    def test_tfidf_logreg_pipeline_uses_explainable_parameters(self):
        model = build_tfidf_logreg_pipeline(
            {
                "candidate": "unit",
                "ngram_range": (1, 2),
                "min_df": 1,
                "max_df": 0.95,
                "sublinear_tf": True,
                "lowercase": True,
                "stop_words": None,
                "C": 0.5,
                "class_weight": "balanced",
                "solver": "lbfgs",
            },
            seed=7,
        )

        self.assertEqual(model.named_steps["tfidf"].ngram_range, (1, 2))
        self.assertTrue(model.named_steps["tfidf"].sublinear_tf)
        self.assertEqual(model.named_steps["clf"].C, 0.5)
        self.assertEqual(model.named_steps["clf"].class_weight, "balanced")
        self.assertEqual(model.named_steps["clf"].solver, "lbfgs")

    def test_best_candidate_prefers_macro_f1_then_accuracy_then_simpler_rank(self):
        dev_results = pd.DataFrame(
            [
                {"candidate": "larger", "macro_f1": 0.60, "accuracy": 0.80, "candidate_rank": 2},
                {"candidate": "stronger", "macro_f1": 0.65, "accuracy": 0.70, "candidate_rank": 3},
                {"candidate": "simpler", "macro_f1": 0.65, "accuracy": 0.70, "candidate_rank": 1},
            ]
        )

        best = best_candidate_from_dev_results(dev_results)

        self.assertEqual(best["candidate"], "simpler")

    def test_summarize_misclassification_patterns_counts_error_pairs(self):
        split = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d"],
                "label": ["dialogue", "dialogue", "mixed", "third_person"],
            }
        )

        patterns = summarize_misclassification_patterns(
            split,
            predictions=["mixed", "mixed", "dialogue", "third_person"],
            top_n=3,
        )

        self.assertEqual(
            patterns.to_dict("records"),
            [
                {
                    "gold": "dialogue",
                    "prediction": "mixed",
                    "count": 2,
                    "gold_total": 2,
                    "share_of_gold": 1.0,
                },
                {
                    "gold": "mixed",
                    "prediction": "dialogue",
                    "count": 1,
                    "gold_total": 1,
                    "share_of_gold": 1.0,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
