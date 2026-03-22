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


LABEL_COL = "label"
TEXT_COL = "text"
RANDOM_STATE = 42


def normalize_text(text: str) -> str:
    text = str(text)
    text = (
        text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    )  # \r\n：Windows 风格换行 \n：Unix/Linux/macOS 常见换行 \r：老式 Mac 换行
    text = (
        text.strip().lower()
    )  # 去掉字符串开头和结尾的空白字符，例如空格、换行、制表符。
    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # keep only letters, digits, and whitespace
    # punctuation is removed here
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)  # 把连续多个空白字符替换成一个空格。
    return text


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df[TEXT_COL] = df[TEXT_COL].apply(
        normalize_text
    )  # 对于TTEXT_COL这一列中的每一个元素进行nnormalize_text
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(
        drop=True
    )  # 删除在这两列里有缺失值的行。
    return df


def split(df: pd.DataFrame):
    n = len(df)
    mid = n // 2
    train_pool = df.iloc[:mid].copy().reset_index(drop=True)
    test_df = df.iloc[mid:].copy().reset_index(drop=True)
    return train_pool, test_df


# 八股 1 ： embedding and vectorizer difference
def train_and_evaluate(
    train_t, train_l, test_t, test_l, vectorizer, model_name, balance=False
):
    X_train = vectorizer.fit_transform(
        train_t
    )  # fit会建立一个dict，里面是单词：数字。然后根据这个词典进行train_text的向量化
    X_test = vectorizer.transform(test_t)  # 使用完全一样的词典，测试集不能参与训练
    # 八股 2 : Where the old dict store? => dict = vocabulary_，在原来vvectorizer里面

    if balance:
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    else:
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X_train, train_l)  # train

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)

    acc = accuracy_score(test_l, pred)
    macro_f1 = f1_score(test_l, pred, average="macro")
    # 八股 3 : 为什么这里使用macro而不是micro?
    # 样本不均衡（Class Imbalance）

    print(model_name)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("Classification Report:")
    print(
        classification_report(test_l, pred, zero_division=0)
    )  # precision \ recall \ f1-score \ support

    print("Confusion Matrix:")
    print(confusion_matrix(test_l, pred, labels=clf.classes_))
    # 八股 4: clf.class_的作用是什么？
    # 模型在训练后自己学到的内部变量
    print("Labels order:", list(clf.classes_))

    return clf, vectorizer, pred, prob, acc, macro_f1


def TopFPC(clf, vectorizer, top_k=10):
    feature_names = np.array(vectorizer.get_feature_names_out())  # dict => features
    classes = clf.classes_

    print("Top Features Per Class")
    for i, label in enumerate(classes):
        coefs = clf.coef_[
            i
        ]  # coef_ => weight : array of shape (n_features, ) or (n_targets, n_features)
        top_idx = np.argsort(coefs)[-top_k:][::-1]
        top_words = feature_names[top_idx]
        top_weights = coefs[top_idx]

        print(f"\nClass: {label}")
        for word, weight in zip(top_words, top_weights):
            print(f"{word:20s} {weight:.4f}")


def TopConErr(test_texts, true_labels, pred_labels, prob, classes, top_k=3):
    errors = []
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for text, true, pred, row_prob in zip(test_texts, true_labels, pred_labels, prob):
        if true != pred:
            pred_idx = class_to_idx[pred]  # 找到pred的index
            confidence = row_prob[pred_idx]  # 根据index找到当前class label
            errors.append((confidence, text, true, pred))

    errors.sort(key=lambda x: x[0], reverse=True)  # 根据conconfidence进行sort, 从大到小

    print("Top Confident Mistakes")
    for i, (confidence, text, true, pred) in enumerate(errors[:top_k], start=1):
        print(f"\nMistake #{i}")
        print(f"Text: {text}")
        print(f"Gold: {true}")
        print(f"Pred: {pred}")
        print(f"Confidence: {confidence:.4f}")


def main():
    df = load_data("../phase1/final.csv")
    train_df, test_df = split(df)

    train_texts = train_df[TEXT_COL].tolist()
    train_labels = train_df[LABEL_COL].tolist()
    test_texts = test_df[TEXT_COL].tolist()
    test_labels = test_df[LABEL_COL].tolist()

    # Baseline model
    bow_vectorizer = CountVectorizer(
        binary=True,  # Only care about attendance
        lowercase=False,  # lower() in nnormalize_text
        token_pattern=r"(?u)\b\w+\b",  # (?u) => unicode, \b => 单词边界, \w+ => 单词/数字/下划线， \b => 单词边界
    )

    bow_clf, bow_vec, bow_pred, bow_prob, bow_acc, bow_macro_f1 = train_and_evaluate(
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        bow_vectorizer,
        "Baseline: Binary BOW + Logistic Regression",
    )

    TopFPC(bow_clf, bow_vec, top_k=10)
    TopConErr(test_texts, test_labels, bow_pred, bow_prob, bow_clf.classes_, top_k=3)
    print("=" * 120)

    # Baseline model + balance
    bow_vectorizer = CountVectorizer(
        binary=True,  # Only care about attendance
        lowercase=False,  # lower() in nnormalize_text
        token_pattern=r"(?u)\b\w+\b",  # (?u) => unicode, \b => 单词边界, \w+ => 单词/数字/下划线， \b => 单词边界
    )

    bow_clf_b, bow_vec_b, bow_pred_b, bow_prob_b, bow_acc_b, bow_macro_f1_b = (
        train_and_evaluate(
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            bow_vectorizer,
            "Baseline: Binary BOW + Logistic Regression + balanced",
            balance=True,
        )
    )

    TopFPC(bow_clf_b, bow_vec_b, top_k=10)
    TopConErr(
        test_texts, test_labels, bow_pred_b, bow_prob_b, bow_clf_b.classes_, top_k=3
    )
    print("=" * 120)

    # Improved model (Word Frequency)
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # ngram
        min_df=2,  # min Frequency
        sublinear_tf=True,  # smooth
        lowercase=False,  # lower() in nnormalize_text
    )

    imp_clf, imp_vec, imp_pred, imp_prob, imp_acc, imp_macro_f1 = train_and_evaluate(
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        tfidf_vectorizer,
        "Improved: TF-IDF (1,2)-gram + Logistic Regression",
    )

    TopFPC(imp_clf, imp_vec, top_k=10)
    TopConErr(test_texts, test_labels, imp_pred, imp_prob, imp_clf.classes_, top_k=3)
    print("=" * 120)

    print("Results Table")
    print(f"{'Model':45s} {'Accuracy':>10s} {'Macro-F1':>10s}")
    print(f"{'Baseline: Binary BOW + LogReg':45s} {bow_acc:10.4f} {bow_macro_f1:10.4f}")
    print(
        f"{'Baseline: Improved Binary BOW + LogReg + balanced':45s} {bow_acc_b:10.4f} {bow_macro_f1_b:10.4f}"
    )
    print(
        f"{'Improved: TF-IDF (1,2)-gram + LogReg':45s} {imp_acc:10.4f} {imp_macro_f1:10.4f}"
    )


if __name__ == "__main__":
    main()
