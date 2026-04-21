"""Normalize Brahe-Novels enunciation labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LABELS = {"first_person", "third_person", "dialogue", "mixed", "epistolary", "other"}

DIRECT_ALIASES = {
    "dialog": "dialogue",
    "dialogue": "dialogue",
    "dialogues": "dialogue",
    "conversation": "dialogue",
    "conversational dialogue": "dialogue",
    "third person narrative": "third_person",
    "third person narration": "third_person",
    "third person narrator": "third_person",
    "third person omniscient narrator": "third_person",
    "the text is written in the third person narrative": "third_person",
    "third person description": "third_person",
    "scholarly third person narrative": "third_person",
    "omniscient narrator": "third_person",
    "first person narrative": "first_person",
    "first person narration": "first_person",
    "first person narrator": "first_person",
}

EPISTOLARY_TERMS = (
    "epistolary",
    "letter",
    "letters",
    "diary",
    "journal",
    "correspondence",
    "memoir",
)
DIALOGUE_TERMS = (
    "dialogue",
    "dialog",
    "conversation",
    "conversational",
    "direct speech",
    "characters speaking",
    "multiple characters speaking",
    "various characters speaking",
)
WEAK_DIALOGUE_TERMS = (
    "speaking",
    "speech",
)
FIRST_PERSON_TERMS = ("first person", "1st person", "i narrator", "i-narrator")
THIRD_PERSON_TERMS = (
    "third person",
    "3rd person",
    "omniscient narrator",
    "third party",
)
MIXED_TERMS = ("mixed", "multiple perspectives", "alternating", "combination")
NARRATIVE_TERMS = ("narrative", "narration", "narrator", "description")


def normalize_enunciation(raw: str | None) -> dict[str, Any]:
    """Map an ``enunciation_raw`` value to a normalized label and confidence flag."""
    if raw is None or not str(raw).strip():
        return _result(raw, "other", True, "missing enunciation_raw")

    cleaned = clean_value(raw)
    direct = DIRECT_ALIASES.get(cleaned)
    if direct:
        return _result(raw, direct, False, f"direct alias: {cleaned}")

    has_epistolary = _contains_any(cleaned, EPISTOLARY_TERMS)
    has_dialogue = _contains_any(cleaned, DIALOGUE_TERMS)
    has_weak_dialogue = _contains_any(cleaned, WEAK_DIALOGUE_TERMS)
    has_first = _contains_any(cleaned, FIRST_PERSON_TERMS)
    has_third = _contains_any(cleaned, THIRD_PERSON_TERMS)
    has_mixed = _contains_any(cleaned, MIXED_TERMS)
    has_narrative = _contains_any(cleaned, NARRATIVE_TERMS)

    if has_epistolary:
        return _result(raw, "epistolary", False, "mentions epistolary form")

    if has_mixed or (has_first and has_third) or (has_dialogue and has_narrative):
        return _result(raw, "mixed", False, "combines narrative modes")

    if has_dialogue:
        return _result(raw, "dialogue", False, "mentions dialogue or speech")

    if has_weak_dialogue:
        return _result(raw, "dialogue", True, "mentions speech/speaking without explicit dialogue")

    if has_first:
        return _result(raw, "first_person", False, "mentions first-person mode")

    if has_third:
        return _result(raw, "third_person", False, "mentions third-person mode")

    return _result(raw, "other", True, "no first-pass rule matched")


def clean_value(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_classification_csv(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        for record in records:
            if not is_clean_classification_record(record):
                continue
            label = record["enunciation_norm"]
            writer.writerow({"text": record.get("full_text") or "", "label": label})
            count += 1
    return count


def write_frequencies(counter: Counter[str], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["enunciation_raw", "count"])
        for value, count in counter.most_common():
            writer.writerow([value, count])


def write_uncertain(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["instruction_id", "enunciation_raw", "enunciation_norm", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if record["enunciation_uncertain"]:
                writer.writerow(
                    {
                        "instruction_id": record.get("instruction_id"),
                        "enunciation_raw": record.get("enunciation_raw"),
                        "enunciation_norm": record.get("enunciation_norm"),
                        "reason": record.get("enunciation_norm_reason"),
                    }
                )
                count += 1
    return count


def normalize_records(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for record in records:
        normalized = normalize_enunciation(record.get("enunciation_raw"))
        next_record = dict(record)
        next_record.update(normalized)
        yield next_record


def is_clean_classification_record(record: dict[str, Any]) -> bool:
    return record.get("enunciation_norm") != "other" and not record.get(
        "enunciation_uncertain", True
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _result(raw: str | None, label: str, uncertain: bool, reason: str) -> dict[str, Any]:
    if label not in LABELS:
        raise ValueError(f"unknown normalized label: {label}")
    return {
        "enunciation_raw": raw,
        "enunciation_norm": label,
        "enunciation_uncertain": uncertain,
        "enunciation_norm_reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/brahe_novels_parsed.jsonl")
    parser.add_argument("--output", default="data/brahe_novels_enunciation_normalized.jsonl")
    parser.add_argument(
        "--classification-output",
        default="data/brahe_enunciation_classification.csv",
    )
    parser.add_argument(
        "--frequency-output",
        default="data/enunciation_raw_frequencies.csv",
    )
    parser.add_argument(
        "--uncertain-output",
        default="data/enunciation_uncertain_cases.csv",
    )
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    frequencies = Counter(row.get("enunciation_raw") or "" for row in rows)
    normalized_rows = list(normalize_records(rows))
    label_counts = Counter(row["enunciation_norm"] for row in normalized_rows)

    write_frequencies(frequencies, args.frequency_output)
    normalized_count = write_jsonl(normalized_rows, args.output)
    classification_count = write_classification_csv(normalized_rows, args.classification_output)
    uncertain_count = write_uncertain(normalized_rows, args.uncertain_output)

    print(f"Wrote {normalized_count} normalized records to {args.output}")
    print(f"Wrote {classification_count} classification rows to {args.classification_output}")
    print(f"Wrote raw value frequencies to {args.frequency_output}")
    print(f"Wrote {uncertain_count} uncertain cases to {args.uncertain_output}")
    print("Most common raw values:")
    for value, count in frequencies.most_common(20):
        print(f"{count}\t{value!r}")
    print("Normalized label counts:")
    for label, count in label_counts.most_common():
        print(f"{label}\t{count}")
    print("Before/after examples:")
    for row in normalized_rows[: args.examples]:
        print(
            json.dumps(
                {
                    "enunciation_raw": row.get("enunciation_raw"),
                    "enunciation_norm": row.get("enunciation_norm"),
                    "enunciation_uncertain": row.get("enunciation_uncertain"),
                    "reason": row.get("enunciation_norm_reason"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
