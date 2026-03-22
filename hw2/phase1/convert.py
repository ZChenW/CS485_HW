import json
import csv
import re

IN_PATH = "group13.json"
OUT_PATH = "sheet.csv"


def normalize_sentence(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ")  # 关键：去掉内部换行
    s = " ".join(s.split())  # 压缩多余空格
    return s


def extract_sentences(data):
    sentences = []

    # list: ["...", ...] 或 [{"txt": "..."}...]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                sentences.append(item)
            elif isinstance(item, dict):
                for key in ["txt", "text", "sentence", "content"]:
                    if key in item and isinstance(item[key], str):
                        sentences.append(item[key])
                        break
        return sentences

    # dict
    if isinstance(data, dict):
        # 常见：{"txt": [...]} 或 {"txt": {"0": "..."}}
        if "txt" in data:
            return extract_sentences(data["txt"])

        # 另一种：{"0": "...", "1": "...", ...} + 可能夹杂别的key
        numeric_items = []
        for k, v in data.items():
            if isinstance(v, str) and re.fullmatch(r"\d+", str(k)):
                numeric_items.append((int(k), v))
        if numeric_items:
            numeric_items.sort(key=lambda x: x[0])
            return [v for _, v in numeric_items]

    return sentences  # fallback: empty


with open(IN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = extract_sentences(data)

with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    # 表头：你要的三列（Text/Label/Notes）+ 第一列 id
    w.writerow(["id", "Text", "Label", "Notes"])
    for i, s in enumerate(sentences):
        w.writerow([i, normalize_sentence(s), "", ""])

print(f"OK: wrote {len(sentences)} rows to {OUT_PATH}")
