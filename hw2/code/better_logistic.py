import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

LABEL_COL = "label"
TEXT_COL = "text"
RANDOM_STATE = 42
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def normalize_text(text: str) -> str:
    text = str(text)

    # normalize line breaks
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # keep only letters, digits, and whitespace
    # punctuation is removed here
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [stemmer.stem(tok) for tok in tokens]
    # tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    text = " ".join(tokens)

    return text


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    df[TEXT_COL] = df[TEXT_COL].apply(normalize_text)
    return df


def split_train_test(df: pd.DataFrame):
    """
    Keep the homework's original spirit:
    first half = train pool
    second half = test
    Then we split the train pool again into train/dev for tuning.
    """
    n = len(df)
    mid = n // 2
    train_pool = df.iloc[:mid].copy().reset_index(drop=True)
    test_df = df.iloc[mid:].copy().reset_index(drop=True)
    return train_pool, test_df


def make_cv(y, max_splits=5):
    counts = Counter(y)
    min_class_count = min(counts.values())

    # 至少要 2-fold，至多不能超过最小类样本数
    n_splits = max(2, min(max_splits, min_class_count))

    print(f"Using {n_splits}-fold CV (smallest class count = {min_class_count})")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def print_top_features(clf, vectorizer, top_k=10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = clf.classes_

    print("Top Features Per Class")
    for i, label in enumerate(classes):
        coefs = clf.coef_[i]
        top_idx = np.argsort(coefs)[-top_k:][::-1]
        top_words = feature_names[top_idx]
        top_weights = coefs[top_idx]

        print(f"\nClass: {label}")
        for word, weight in zip(top_words, top_weights):
            print(f"{word:20s} {weight:.4f}")


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


def tune_baseline_binary_bow(X_train, y_train):
    """
    Baseline:
    Binary BOW + Logistic Regression

    We still tune a few classifier hyperparameters.
    """
    pipeline = Pipeline(
        [
            (
                "vect",
                CountVectorizer(
                    binary=True,
                    lowercase=False,  # already done in normalize_text
                    stop_words="english",
                    token_pattern=r"(?u)\b[a-z][a-z0-9]+\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "vect__min_df": [1, 2, 3],
        "vect__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
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
    print("Best Baseline Params:")
    print(grid.best_params_)
    print(f"Best CV Macro-F1: {grid.best_score_:.4f}")

    return grid


def tune_tfidf_model(X_train, y_train):
    """
    Improved model:
    TF-IDF + L2 normalization + Logistic Regression
    """
    pipeline = Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(
                    lowercase=False,  # already done in normalize_text
                    stop_words="english",
                    token_pattern=r"(?u)\b[a-z][a-z0-9]+\b",
                    sublinear_tf=True,
                    norm="l2",  # explicit normalization
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "vect__min_df": [1, 2, 3],
        "clf__C": [0.1, 1.0, 10.0],
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
    print("Best Improved Model Params:")
    print(grid.best_params_)
    print(f"Best CV Macro-F1: {grid.best_score_:.4f}")

    return grid


def main():
    df = load_data("../phase1/final.csv")

    # first half: train pool, second half: held-out test
    train_pool_df, test_df = split_train_test(df)

    # split the train pool again for a dev set
    train_df, dev_df = train_test_split(
        train_pool_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_pool_df[LABEL_COL],
    )

    X_train = train_df[TEXT_COL].tolist()
    y_train = train_df[LABEL_COL].tolist()

    X_dev = dev_df[TEXT_COL].tolist()
    y_dev = dev_df[LABEL_COL].tolist()

    X_test = test_df[TEXT_COL].tolist()
    y_test = test_df[LABEL_COL].tolist()

    print(f"Train size: {len(train_df)}")
    print(f"Dev size:   {len(dev_df)}")
    print(f"Test size:  {len(test_df)}")

    # -----------------------------
    # Baseline tuning
    # -----------------------------
    baseline_grid = tune_baseline_binary_bow(X_train, y_train)

    print("\nDev-set performance for tuned baseline:")
    baseline_dev_pred, baseline_dev_prob, baseline_dev_acc, baseline_dev_f1 = (
        evaluate_model(
            baseline_grid.best_estimator_, X_dev, y_dev, "Tuned Baseline on Dev"
        )
    )

    # retrain tuned baseline on full train pool (train + dev), then evaluate on test
    best_baseline = baseline_grid.best_estimator_
    best_baseline.fit(
        train_pool_df[TEXT_COL].tolist(), train_pool_df[LABEL_COL].tolist()
    )

    baseline_test_pred, baseline_test_prob, baseline_test_acc, baseline_test_f1 = (
        evaluate_model(best_baseline, X_test, y_test, "Final Tuned Baseline on Test")
    )

    baseline_vect = best_baseline.named_steps["vect"]
    baseline_clf = best_baseline.named_steps["clf"]
    print_top_features(baseline_clf, baseline_vect, top_k=10)
    print_top_confident_errors(
        X_test,
        y_test,
        baseline_test_pred,
        baseline_test_prob,
        baseline_clf.classes_,
        top_k=3,
    )

    # -----------------------------
    # Improved model tuning
    # -----------------------------
    improved_grid = tune_tfidf_model(X_train, y_train)

    print("\nDev-set performance for tuned improved model:")
    improved_dev_pred, improved_dev_prob, improved_dev_acc, improved_dev_f1 = (
        evaluate_model(
            improved_grid.best_estimator_, X_dev, y_dev, "Tuned Improved Model on Dev"
        )
    )

    # retrain tuned improved model on full train pool (train + dev), then evaluate on test
    best_improved = improved_grid.best_estimator_
    best_improved.fit(
        train_pool_df[TEXT_COL].tolist(), train_pool_df[LABEL_COL].tolist()
    )

    improved_test_pred, improved_test_prob, improved_test_acc, improved_test_f1 = (
        evaluate_model(
            best_improved, X_test, y_test, "Final Tuned Improved Model on Test"
        )
    )

    improved_vect = best_improved.named_steps["vect"]
    improved_clf = best_improved.named_steps["clf"]
    print_top_features(improved_clf, improved_vect, top_k=10)
    print_top_confident_errors(
        X_test,
        y_test,
        improved_test_pred,
        improved_test_prob,
        improved_clf.classes_,
        top_k=3,
    )

    # -----------------------------
    # Final summary
    # -----------------------------
    print("=" * 120)
    print("Results Table")
    print(f"{'Model':40s} {'Accuracy':>10s} {'Macro-F1':>10s}")
    print(
        f"{'Final Tuned Baseline':40s} {baseline_test_acc:10.4f} {baseline_test_f1:10.4f}"
    )
    print(
        f"{'Final Tuned Improved TF-IDF':40s} {improved_test_acc:10.4f} {improved_test_f1:10.4f}"
    )


if __name__ == "__main__":
    main()
