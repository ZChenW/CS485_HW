"""Reproducible text classification baselines for Brahe datasets."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_DATASETS = {
    "enunciation": "data/brahe_enunciation_classification.csv",
    "genre": "data/brahe_genre_classification.csv",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


def load_classification_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df = df[(df["text"].str.strip() != "") & (df["label"].str.strip() != "")]
    return df.reset_index(drop=True)


def make_stratified_splits(
    df: pd.DataFrame,
    seed: int,
    train_size: float = 0.70,
    dev_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if round(train_size + dev_size + test_size, 8) != 1.0:
        raise ValueError("train/dev/test proportions must sum to 1.0")
    counts = df["label"].value_counts()
    if (counts < 4).any():
        small = counts[counts < 4].to_dict()
        raise ValueError(
            f"each label needs at least 4 rows for stratified 70/15/15 splits: {small}"
        )

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )
    relative_test_size = test_size / (dev_size + test_size)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_df["label"],
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        dev_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def train_majority_baseline(train_df: pd.DataFrame) -> DummyClassifier:
    model = DummyClassifier(strategy="most_frequent")
    model.fit(train_df["text"], train_df["label"])
    return model


def train_tfidf_logreg(train_df: pd.DataFrame, seed: int) -> Pipeline:
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(train_df["text"], train_df["label"])
    return model


def evaluate_predictions(
    split_df: pd.DataFrame,
    predictions: Iterable[str],
    labels: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]], pd.DataFrame]:
    y_true = split_df["label"].tolist()
    y_pred = list(predictions)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    misclassified = split_df.loc[
        pd.Series(y_true) != pd.Series(y_pred), ["text", "label"]
    ].copy()
    misclassified["prediction"] = [
        pred for gold, pred in zip(y_true, y_pred, strict=True) if gold != pred
    ]
    return metrics, report, misclassified.reset_index(drop=True)


def save_confusion_matrix(
    split_df: pd.DataFrame,
    predictions: Iterable[str],
    labels: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    matrix = confusion_matrix(split_df["label"], list(predictions), labels=labels)
    fig_width = max(7, min(16, 0.75 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_dataset(spec: DatasetSpec, output_root: str | Path, seed: int) -> pd.DataFrame:
    dataset_dir = Path(output_root) / spec.name
    split_dir = dataset_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    df = load_classification_csv(spec.path)
    train_df, dev_df, test_df = make_stratified_splits(df, seed=seed)
    train_df.to_csv(split_dir / "train.csv", index=False)
    dev_df.to_csv(split_dir / "dev.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    labels = sorted(train_df["label"].unique())
    (dataset_dir / "labels.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8"
    )

    models = {
        "majority": train_majority_baseline(train_df),
        "tfidf_logreg": train_tfidf_logreg(train_df, seed=seed),
    }

    summary_rows = []
    for model_name, model in models.items():
        model_dir = dataset_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")

        for split_name, split_df in {"dev": dev_df, "test": test_df}.items():
            predictions = model.predict(split_df["text"])
            metrics, report, misclassified = evaluate_predictions(
                split_df, predictions, labels
            )
            metrics_with_context = {
                "dataset": spec.name,
                "model": model_name,
                "split": split_name,
                **metrics,
            }
            summary_rows.append(metrics_with_context)

            _write_json(model_dir / f"{split_name}_metrics.json", metrics_with_context)
            _write_json(model_dir / f"{split_name}_classification_report.json", report)
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
            misclassified.to_csv(
                model_dir / f"{split_name}_misclassified.csv", index=False
            )
            save_confusion_matrix(
                split_df,
                predictions,
                labels,
                model_dir / f"{split_name}_confusion_matrix.png",
                title=f"{spec.name} {model_name} {split_name}",
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(dataset_dir / "summary_metrics.csv", index=False)
    return summary


def parse_dataset_specs(values: list[str] | None) -> list[DatasetSpec]:
    if not values:
        return [
            DatasetSpec(name, Path(path)) for name, path in DEFAULT_DATASETS.items()
        ]

    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"dataset must be NAME=PATH, got: {value}")
        name, path = value.split("=", 1)
        specs.append(DatasetSpec(name=name.strip(), path=Path(path.strip())))
    return specs


def _write_json(path: str | Path, data: object) -> None:
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset spec as NAME=PATH. Repeat to run multiple. Defaults to enunciation and genre.",
    )
    parser.add_argument("--output-root", default="outputs/baselines")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    specs = parse_dataset_specs(args.dataset)
    all_summaries = []
    for spec in specs:
        print(f"Running baselines for {spec.name}: {spec.path}")
        summary = run_dataset(spec, output_root=args.output_root, seed=args.seed)
        print(summary.to_string(index=False))
        all_summaries.append(summary)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.concat(all_summaries, ignore_index=True).to_csv(
        output_root / "all_summary_metrics.csv",
        index=False,
    )
    print(f"Wrote combined summary to {output_root / 'all_summary_metrics.csv'}")


if __name__ == "__main__":
    main()
