import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABEL_COL = "label"
TEXT_COL = "text"
RANDOM_STATE = 42


def normalize_text_for_embeddings(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # keep apostrophes out; just basic token cleanup for glove matching
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_embeddings(text: str):
    return [tok for tok in normalize_text_for_embeddings(text).split() if tok]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    return df


def split_train_test(df: pd.DataFrame):
    n = len(df)
    mid = n // 2
    train_pool = df.iloc[:mid].copy().reset_index(drop=True)
    test_df = df.iloc[mid:].copy().reset_index(drop=True)
    return train_pool, test_df


def make_cv(y, max_splits=5):
    counts = Counter(y)
    min_class_count = min(counts.values())
    n_splits = max(2, min(max_splits, min_class_count))
    print(f"Using {n_splits}-fold CV (smallest class count = {min_class_count})")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def load_glove(path: str, max_vocab=None):
    embeddings = {}
    dim = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if dim is None:
                dim = len(vec)
            embeddings[word] = vec
            if max_vocab is not None and (i + 1) >= max_vocab:
                break
    if dim is None:
        raise ValueError(f"No embeddings could be read from: {path}")
    print(f"Loaded {len(embeddings)} embeddings with dim={dim}")
    return embeddings, dim


def average_embedding(tokens, glove, dim):
    vecs = [glove[tok] for tok in tokens if tok in glove]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def build_embedding_matrix(texts, glove, dim, report_prefix=""):
    X = np.zeros((len(texts), dim), dtype=np.float32)
    total_tokens = 0
    covered_tokens = 0
    zero_docs = 0

    for i, text in enumerate(texts):
        tokens = tokenize_for_embeddings(text)
        total_tokens += len(tokens)
        covered_tokens += sum(tok in glove for tok in tokens)
        X[i] = average_embedding(tokens, glove, dim)
        if not np.any(X[i]):
            zero_docs += 1

    coverage = (covered_tokens / total_tokens) if total_tokens else 0.0
    print(
        f"{report_prefix}token coverage={coverage:.4f} "
        f"({covered_tokens}/{total_tokens}), zero-vector docs={zero_docs}/{len(texts)}"
    )
    return X


def evaluate_model(model, X_test, y_test, model_name):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    print("=" * 120)
    print(model_name)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred, labels=model.classes_))
    print("Labels order:", list(model.classes_))

    return pred, prob, acc, macro_f1


def print_top_confident_errors(
    test_texts, true_labels, pred_labels, prob, classes, top_k=3
):
    errors = []
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for text, true, pred, row_prob in zip(test_texts, true_labels, pred_labels, prob):
        if true != pred:
            pred_idx = class_to_idx[pred]
            confidence = row_prob[pred_idx]
            errors.append((confidence, text, true, pred))

    errors.sort(key=lambda x: x[0], reverse=True)

    print("Top Confident Mistakes")
    for i, (confidence, text, true, pred) in enumerate(errors[:top_k], start=1):
        print(f"\nMistake #{i}")
        print(f"Text: {text}")
        print(f"Gold: {true}")
        print(f"Pred: {pred}")
        print(f"Confidence: {confidence:.4f}")


def tune_glove_logreg(X_train, y_train):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }

    cv = make_cv(y_train)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print("=" * 120)
    print("Best GloVe Averaging Model Params:")
    print(grid.best_params_)
    print(f"Best CV Macro-F1: {grid.best_score_:.4f}")
    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Homework 4.3: averaged GloVe embeddings + Logistic Regression"
    )
    parser.add_argument(
        "--data", default="../phase1/final.csv", help="Path to final.csv"
    )
    parser.add_argument(
        "--glove",
        required=True,
        help="Path to a GloVe txt file, e.g. glove.twitter.27B.50d.txt",
    )
    parser.add_argument(
        "--max-vocab", type=int, default=None, help="Optional cap for debugging"
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    if not os.path.exists(args.glove):
        raise FileNotFoundError(f"GloVe file not found: {args.glove}")

    df = load_data(args.data)
    glove, dim = load_glove(args.glove, max_vocab=args.max_vocab)

    train_pool_df, test_df = split_train_test(df)

    train_df, dev_df = train_test_split(
        train_pool_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_pool_df[LABEL_COL],
    )

    X_train_text = train_df[TEXT_COL].tolist()
    y_train = train_df[LABEL_COL].tolist()

    X_dev_text = dev_df[TEXT_COL].tolist()
    y_dev = dev_df[LABEL_COL].tolist()

    X_test_text = test_df[TEXT_COL].tolist()
    y_test = test_df[LABEL_COL].tolist()

    print(f"Train size: {len(train_df)}")
    print(f"Dev size:   {len(dev_df)}")
    print(f"Test size:  {len(test_df)}")

    X_train = build_embedding_matrix(X_train_text, glove, dim, report_prefix="Train: ")
    X_dev = build_embedding_matrix(X_dev_text, glove, dim, report_prefix="Dev:   ")
    X_test = build_embedding_matrix(X_test_text, glove, dim, report_prefix="Test:  ")

    glove_grid = tune_glove_logreg(X_train, y_train)

    print("\nDev-set performance for tuned GloVe model:")
    evaluate_model(
        glove_grid.best_estimator_, X_dev, y_dev, "Tuned GloVe Averaging Model on Dev"
    )

    X_train_pool = build_embedding_matrix(
        train_pool_df[TEXT_COL].tolist(), glove, dim, report_prefix="TrainPool: "
    )
    best_glove = glove_grid.best_estimator_
    best_glove.fit(X_train_pool, train_pool_df[LABEL_COL].tolist())

    test_pred, test_prob, test_acc, test_f1 = evaluate_model(
        best_glove, X_test, y_test, "Final Tuned GloVe Averaging Model on Test"
    )

    print_top_confident_errors(
        X_test_text,
        y_test,
        test_pred,
        test_prob,
        best_glove.named_steps["clf"].classes_,
        top_k=3,
    )

    print("=" * 120)
    print("Results Table")
    print(f"{'Model':45s} {'Accuracy':>10s} {'Macro-F1':>10s}")
    print(
        f"{'Final Tuned GloVe Averaging + LogReg':45s} {test_acc:10.4f} {test_f1:10.4f}"
    )


if __name__ == "__main__":
    main()
