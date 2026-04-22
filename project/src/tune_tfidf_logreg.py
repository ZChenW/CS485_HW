"""Practical TF-IDF + Logistic Regression tuning and analysis.

This script keeps the original baseline intact and runs a small dev-set search
over explainable classical hyperparameters. It writes comparison metrics and
error-analysis artifacts for the enunciation and genre classification tasks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from src.baseline_classification import (
    DEFAULT_DATASETS,
    DatasetSpec,
    evaluate_predictions,
    load_classification_csv,
    make_stratified_splits,
    parse_dataset_specs,
    save_confusion_matrix,
    train_majority_baseline,
    train_tfidf_logreg,
)


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


Candidate = dict[str, object]


def tuning_candidates(task_name: str) -> list[Candidate]:
    """Return a small, report-friendly candidate list.

    The same search space is used for both tasks so the comparison is easy to
    explain. English stop words are intentionally excluded because the excerpts
    are multilingual and literary function words can be label-informative.
    """
    candidates: list[Candidate] = [
        {
            "candidate": "original_settings",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 1.0,
            "sublinear_tf": False,
            "lowercase": True,
            "stop_words": None,
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        {
            "candidate": "unigrams_balanced",
            "ngram_range": (1, 1),
            "min_df": 2,
            "max_df": 1.0,
            "sublinear_tf": True,
            "lowercase": True,
            "stop_words": None,
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        {
            "candidate": "bigrams_sublinear_C1",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 1.0,
            "sublinear_tf": True,
            "lowercase": True,
            "stop_words": None,
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        {
            "candidate": "bigrams_sublinear_C0_5",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 1.0,
            "sublinear_tf": True,
            "lowercase": True,
            "stop_words": None,
            "C": 0.5,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        {
            "candidate": "bigrams_sublinear_C2",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 1.0,
            "sublinear_tf": True,
            "lowercase": True,
            "stop_words": None,
            "C": 2.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        {
            "candidate": "bigrams_min3_sublinear",
            "ngram_range": (1, 2),
            "min_df": 3,
            "max_df": 1.0,
            "sublinear_tf": True,
            "lowercase": True,
            "stop_words": None,
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        {
            "candidate": "bigrams_unweighted_C1",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 1.0,
            "sublinear_tf": True,
            "lowercase": True,
            "stop_words": None,
            "C": 1.0,
            "class_weight": None,
            "solver": "lbfgs",
        },
        {
            "candidate": "bigrams_maxdf_0_90",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.90,
            "sublinear_tf": True,
            "lowercase": True,
            "stop_words": None,
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        {
            "candidate": "case_sensitive_bigrams",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 1.0,
            "sublinear_tf": True,
            "lowercase": False,
            "stop_words": None,
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
    ]
    return [{**candidate, "task": task_name} for candidate in candidates]


def build_tfidf_logreg_pipeline(candidate: Candidate, seed: int) -> Pipeline:
    """Build a TF-IDF + multinomial Logistic Regression pipeline."""
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=bool(candidate["lowercase"]),
                    strip_accents="unicode",
                    ngram_range=tuple(candidate["ngram_range"]),  # type: ignore[arg-type]
                    min_df=int(candidate["min_df"]),
                    max_df=float(candidate["max_df"]),
                    sublinear_tf=bool(candidate["sublinear_tf"]),
                    stop_words=candidate["stop_words"],  # type: ignore[arg-type]
                    max_features=50000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=float(candidate["C"]),
                    class_weight=candidate["class_weight"],  # type: ignore[arg-type]
                    solver=str(candidate["solver"]),
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    return model


def tune_on_dev(
    task_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    labels: list[str],
    seed: int,
) -> tuple[Candidate, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    candidates = tuning_candidates(task_name)
    for rank, candidate in enumerate(candidates, start=1):
        print(
            f"[{task_name}] candidate {rank}/{len(candidates)}: "
            f"{candidate['candidate']}",
            flush=True,
        )
        model = build_tfidf_logreg_pipeline(candidate, seed=seed)
        model.fit(train_df["text"], train_df["label"])
        predictions = model.predict(dev_df["text"])
        metrics, _, _ = evaluate_predictions(dev_df, predictions, labels)
        rows.append(
            {
                "candidate_rank": rank,
                **serializable_candidate(candidate),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    dev_results = pd.DataFrame(rows)
    best_row = best_candidate_from_dev_results(dev_results)
    best_candidate = next(
        candidate
        for candidate in candidates
        if candidate["candidate"] == best_row["candidate"]
    )
    return best_candidate, dev_results


def best_candidate_from_dev_results(dev_results: pd.DataFrame) -> dict[str, object]:
    """Select by macro-F1, then accuracy, then simpler earlier candidate."""
    ordered = dev_results.sort_values(
        ["macro_f1", "accuracy", "candidate_rank"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return ordered.iloc[0].to_dict()


def serializable_candidate(candidate: Candidate) -> dict[str, object]:
    result = dict(candidate)
    result["ngram_range"] = list(result["ngram_range"])  # type: ignore[index]
    return result


def confusion_matrix_frame(
    split_df: pd.DataFrame,
    predictions: Iterable[str],
    labels: list[str],
) -> pd.DataFrame:
    matrix = confusion_matrix(split_df["label"], list(predictions), labels=labels)
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "gold_label"
    frame.columns.name = "predicted_label"
    return frame


def summarize_misclassification_patterns(
    split_df: pd.DataFrame,
    predictions: Iterable[str],
    top_n: int | None = 20,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "gold": split_df["label"].tolist(),
            "prediction": list(predictions),
        }
    )
    errors = pred_df[pred_df["gold"] != pred_df["prediction"]]
    if errors.empty:
        return pd.DataFrame(
            columns=["gold", "prediction", "count", "gold_total", "share_of_gold"]
        )

    gold_totals = pred_df["gold"].value_counts().rename("gold_total")
    patterns = (
        errors.groupby(["gold", "prediction"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .merge(gold_totals, left_on="gold", right_index=True)
    )
    patterns["share_of_gold"] = patterns["count"] / patterns["gold_total"]
    patterns = patterns.sort_values(
        ["count", "share_of_gold", "gold", "prediction"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    if top_n is not None:
        patterns = patterns.head(top_n)
    return patterns


def label_distribution_frames(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, split_df in splits.items():
        counts = split_df["label"].value_counts().sort_index()
        total = int(counts.sum())
        for label, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(count),
                    "proportion": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def evaluate_and_save_model(
    dataset_name: str,
    model_name: str,
    model: object,
    splits: dict[str, pd.DataFrame],
    labels: list[str],
    dataset_dir: Path,
) -> list[dict[str, object]]:
    model_dir = dataset_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")

    summary_rows: list[dict[str, object]] = []
    for split_name, split_df in splits.items():
        if split_name == "train":
            continue
        predictions = model.predict(split_df["text"])  # type: ignore[attr-defined]
        metrics, report, misclassified = evaluate_predictions(
            split_df, predictions, labels
        )
        metrics_with_context = {
            "dataset": dataset_name,
            "model": model_name,
            "split": split_name,
            **metrics,
        }
        summary_rows.append(metrics_with_context)

        write_json(model_dir / f"{split_name}_metrics.json", metrics_with_context)
        write_json(model_dir / f"{split_name}_classification_report.json", report)
        pd.DataFrame(report).transpose().to_csv(
            model_dir / f"{split_name}_classification_report.csv"
        )
        pd.DataFrame(
            {
                "text": split_df["text"],
                "label": split_df["label"],
                "prediction": predictions,
            }
        ).to_csv(model_dir / f"{split_name}_predictions.csv", index=False)
        misclassified.to_csv(model_dir / f"{split_name}_misclassified.csv", index=False)
        confusion_matrix_frame(split_df, predictions, labels).to_csv(
            model_dir / f"{split_name}_confusion_matrix.csv"
        )
        save_confusion_matrix(
            split_df,
            predictions,
            labels,
            model_dir / f"{split_name}_confusion_matrix.png",
            title=f"{dataset_name} {model_name} {split_name}",
        )
        summarize_misclassification_patterns(split_df, predictions).to_csv(
            model_dir / f"{split_name}_misclassification_patterns.csv",
            index=False,
        )

    return summary_rows


def run_tuned_dataset(
    spec: DatasetSpec,
    output_root: str | Path,
    seed: int,
) -> pd.DataFrame:
    dataset_dir = Path(output_root) / spec.name
    split_dir = dataset_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    df = load_classification_csv(spec.path)
    train_df, dev_df, test_df = make_stratified_splits(df, seed=seed)
    splits = {
        "all": df,
        "train": train_df,
        "dev": dev_df,
        "test": test_df,
    }
    for split_name, split_df in splits.items():
        if split_name != "all":
            split_df.to_csv(split_dir / f"{split_name}.csv", index=False)

    labels = sorted(train_df["label"].unique())
    write_json(dataset_dir / "labels.json", labels)
    label_distribution_frames(splits).to_csv(
        dataset_dir / "label_distribution.csv", index=False
    )

    best_candidate, dev_results = tune_on_dev(
        spec.name,
        train_df,
        dev_df,
        labels,
        seed=seed,
    )
    dev_results.to_csv(dataset_dir / "tuning_dev_results.csv", index=False)
    write_json(
        dataset_dir / "tuning_dev_results.json",
        dev_results.to_dict(orient="records"),
    )
    write_json(dataset_dir / "best_hyperparameters.json", best_candidate)

    tuned_model = build_tfidf_logreg_pipeline(best_candidate, seed=seed)
    tuned_model.fit(train_df["text"], train_df["label"])

    models = {
        "majority": train_majority_baseline(train_df),
        "tfidf_logreg_original": train_tfidf_logreg(train_df, seed=seed),
        "tfidf_logreg_tuned": tuned_model,
    }

    summary_rows = []
    for model_name, model in models.items():
        summary_rows.extend(
            evaluate_and_save_model(
                spec.name,
                model_name,
                model,
                {"train": train_df, "dev": dev_df, "test": test_df},
                labels,
                dataset_dir,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(dataset_dir / "comparison_metrics_long.csv", index=False)
    make_comparison_table(summary).to_csv(
        dataset_dir / "comparison_table.csv", index=False
    )
    write_analysis_report(spec.name, dataset_dir, best_candidate, summary)
    return summary


def make_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.pivot_table(
        index=["dataset", "model"],
        columns="split",
        values=["accuracy", "macro_f1"],
        aggfunc="first",
    )
    table.columns = [f"{split}_{metric}" for metric, split in table.columns]
    return table.reset_index()


def write_analysis_report(
    dataset_name: str,
    dataset_dir: Path,
    best_candidate: Candidate,
    summary: pd.DataFrame,
) -> None:
    comparison = make_comparison_table(summary)
    label_distribution = pd.read_csv(dataset_dir / "label_distribution.csv")
    tuned_report = pd.read_csv(
        dataset_dir / "tfidf_logreg_tuned" / "test_classification_report.csv",
        index_col=0,
    )
    patterns = pd.read_csv(
        dataset_dir / "tfidf_logreg_tuned" / "test_misclassification_patterns.csv"
    )

    original = comparison[comparison["model"] == "tfidf_logreg_original"].iloc[0]
    tuned = comparison[comparison["model"] == "tfidf_logreg_tuned"].iloc[0]
    macro_delta = float(tuned["test_macro_f1"] - original["test_macro_f1"])

    lines = [
        f"# {dataset_name.title()} TF-IDF Logistic Regression Tuning Analysis",
        "",
        "## Setup",
        "",
        "- Split: stratified 70/15/15 train/dev/test with fixed seed.",
        "- Selection: best dev macro-F1, then dev accuracy, then earlier simpler candidate.",
        "- Model family: TF-IDF features with sklearn LogisticRegression using lbfgs.",
        "- Multiclass formulation: sklearn 1.8 lbfgs uses multinomial loss for multiclass problems.",
        "- Stop words: not used because the corpus is multilingual and function words may carry style or genre signal.",
        "",
        "## Best Hyperparameters",
        "",
        "```json",
        json.dumps(serializable_candidate(best_candidate), indent=2),
        "```",
        "",
        "## Model Comparison",
        "",
        markdown_table(comparison),
        "",
        "## Label Distribution",
        "",
        markdown_table(label_distribution),
        "",
        "## Tuned Test Per-Class Results",
        "",
        markdown_table(tuned_report.reset_index(names="label")),
        "",
        "## Top Tuned Test Misclassification Patterns",
        "",
        markdown_table(patterns.head(15))
        if not patterns.empty
        else "No test-set misclassifications.",
        "",
        "## Targeted Error Notes",
        "",
        targeted_error_notes(dataset_name, patterns),
        "",
        "## Tuning Effect",
        "",
        macro_f1_interpretation(macro_delta),
        "",
        "## Label Redesign Recommendation",
        "",
        label_redesign_recommendation(tuned_report, patterns),
        "",
    ]
    (dataset_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def targeted_error_notes(dataset_name: str, patterns: pd.DataFrame) -> str:
    if patterns.empty:
        return "The tuned model made no test-set errors, so targeted pattern checks are not applicable."

    if dataset_name == "enunciation":
        pairs = [
            ("dialogue", "mixed"),
            ("mixed", "dialogue"),
            ("first_person", "epistolary"),
            ("epistolary", "first_person"),
        ]
        absorber = "third_person"
    else:
        pairs = [
            ("historical", "general_fiction"),
            ("general_fiction", "historical"),
            ("drama", "comedy_satire"),
            ("comedy_satire", "drama"),
            ("mystery_crime", "adventure"),
            ("adventure", "mystery_crime"),
            ("speculative", "general_fiction"),
            ("general_fiction", "speculative"),
        ]
        absorber = "general_fiction"

    lines = []
    for gold, prediction in pairs:
        match = patterns[
            (patterns["gold"] == gold) & (patterns["prediction"] == prediction)
        ]
        if match.empty:
            lines.append(
                f"- `{gold}` -> `{prediction}` was not a top tuned test pattern."
            )
        else:
            row = match.iloc[0]
            lines.append(
                f"- `{gold}` -> `{prediction}` occurred {int(row['count'])} times "
                f"({row['share_of_gold']:.1%} of `{gold}` test examples)."
            )

    absorbed = patterns[patterns["prediction"] == absorber]
    if absorbed.empty:
        lines.append(f"- `{absorber}` did not absorb a top error pattern.")
    else:
        total = int(absorbed["count"].sum())
        sources = ", ".join(
            f"{row['gold']} ({int(row['count'])})" for _, row in absorbed.iterrows()
        )
        lines.append(
            f"- `{absorber}` absorbed {total} top-pattern errors from: {sources}."
        )
    return "\n".join(lines)


def macro_f1_interpretation(delta: float) -> str:
    if delta >= 0.02:
        return f"Tuning meaningfully improved test macro-F1 by {delta:.4f}."
    if delta > 0:
        return f"Tuning produced a small test macro-F1 gain of {delta:.4f}; treat it as modest."
    if delta == 0:
        return "Tuning did not change test macro-F1."
    return f"Tuning reduced test macro-F1 by {abs(delta):.4f}; the original baseline is preferable on test."


def label_redesign_recommendation(
    tuned_report: pd.DataFrame,
    patterns: pd.DataFrame,
) -> str:
    class_rows = tuned_report[
        ~tuned_report.index.isin(["accuracy", "macro avg", "weighted avg"])
    ].copy()
    low_f1 = class_rows[(class_rows["support"] >= 2) & (class_rows["f1-score"] < 0.35)]
    if low_f1.empty:
        return (
            "No label redesign is recommended from this baseline alone. "
            "The remaining errors are better treated as model/data ambiguity checks."
        )

    labels = ", ".join(f"`{label}`" for label in low_f1.index.tolist())
    if patterns.empty:
        return (
            f"Audit the low-F1 labels {labels}, but do not redesign labels solely "
            "from this run because there is no dominant confusion pattern."
        )
    top_pairs = ", ".join(
        f"`{row['gold']}` -> `{row['prediction']}`"
        for _, row in patterns.head(5).iterrows()
    )
    return (
        f"Audit the low-F1 labels {labels}. A redesign or merge should only be "
        f"considered if annotation review confirms that frequent confusions such as "
        f"{top_pairs} reflect unclear category boundaries rather than sparse data."
    )


def write_json(path: str | Path, data: object) -> None:
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    display = display.astype(str)

    headers = [str(column) for column in display.columns]
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "Dataset spec as NAME=PATH. Repeat to run multiple. "
            "Defaults to enunciation and genre."
        ),
    )
    parser.add_argument("--output-root", default="outputs/baseline_tuning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    specs = parse_dataset_specs(args.dataset)
    if not specs:
        specs = [
            DatasetSpec(name, Path(path)) for name, path in DEFAULT_DATASETS.items()
        ]

    all_summaries = []
    for spec in specs:
        print(f"Running TF-IDF/logreg tuning for {spec.name}: {spec.path}")
        summary = run_tuned_dataset(
            spec,
            output_root=args.output_root,
            seed=args.seed,
        )
        print(make_comparison_table(summary).to_string(index=False))
        all_summaries.append(summary)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(output_root / "all_comparison_metrics_long.csv", index=False)
    make_comparison_table(combined).to_csv(
        output_root / "all_comparison_table.csv",
        index=False,
    )
    print(f"Wrote tuning outputs to {output_root}")


if __name__ == "__main__":
    main()
