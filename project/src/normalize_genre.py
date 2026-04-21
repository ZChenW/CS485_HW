"""Normalize Brahe-Novels genre labels for classification."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LABELS = {
    "historical",
    "drama",
    "romance",
    "adventure",
    "poetry",
    "general_fiction",
    "children_young_adult",
    "mystery_crime",
    "comedy_satire",
    "speculative",
    "nonfiction_essay",
    "religious_philosophical",
    "other",
}

LABEL_PRIORITY = [
    "historical",
    "drama",
    "romance",
    "adventure",
    "poetry",
    "general_fiction",
    "children_young_adult",
    "mystery_crime",
    "comedy_satire",
    "speculative",
    "nonfiction_essay",
    "religious_philosophical",
]

DIRECT_ALIASES = {
    "fiction": "general_fiction",
    "fictional narrative": "general_fiction",
    "fictional prose": "general_fiction",
    "literary fiction": "general_fiction",
    "realistic fiction": "general_fiction",
    "domestic fiction": "general_fiction",
    "psychological fiction": "general_fiction",
    "short story": "general_fiction",
    "character study": "general_fiction",
    "slice of life": "general_fiction",
    "historical fiction": "historical",
    "historical novel": "historical",
    "historical non fiction": "historical",
    "historical account": "historical",
    "historical essay": "historical",
    "historical document": "historical",
    "historical narrative": "historical",
    "historical text": "historical",
    "historical analysis": "historical",
    "historical adventure": "historical",
    "drama": "drama",
    "family drama": "drama",
    "historical drama": "drama",
    "psychological drama": "drama",
    "domestic drama": "drama",
    "legal drama": "drama",
    "tragedy": "drama",
    "tragic novel": "drama",
    "tragic romance": "drama",
    "play": "drama",
    "romance": "romance",
    "romantic fiction": "romance",
    "historical romance": "romance",
    "adventure": "adventure",
    "adventure fiction": "adventure",
    "adventure novel": "adventure",
    "western": "adventure",
    "western fiction": "adventure",
    "poetry": "poetry",
    "epic poetry": "poetry",
    "epic poem": "poetry",
    "love poetry": "poetry",
    "children s literature": "children_young_adult",
    "children s fantasy": "children_young_adult",
    "young adult": "children_young_adult",
    "young adult fiction": "children_young_adult",
    "coming of age": "children_young_adult",
    "coming of age novel": "children_young_adult",
    "coming of age story": "children_young_adult",
    "bildungsroman": "children_young_adult",
    "mystery": "mystery_crime",
    "crime fiction": "mystery_crime",
    "detective fiction": "mystery_crime",
    "thriller": "mystery_crime",
    "psychological thriller": "mystery_crime",
    "mystery thriller": "mystery_crime",
    "comedy": "comedy_satire",
    "satire": "comedy_satire",
    "political satire": "comedy_satire",
    "humor": "comedy_satire",
    "fantasy": "speculative",
    "science fiction": "speculative",
    "sci fi": "speculative",
    "gothic fiction": "speculative",
    "gothic novel": "speculative",
    "horror": "speculative",
    "non fiction": "nonfiction_essay",
    "non fiction essay": "nonfiction_essay",
    "essay": "nonfiction_essay",
    "biography": "nonfiction_essay",
    "memoir": "nonfiction_essay",
    "travel literature": "nonfiction_essay",
    "travel writing": "nonfiction_essay",
    "travelogue": "nonfiction_essay",
    "literary criticism": "nonfiction_essay",
    "political essay": "nonfiction_essay",
    "political discourse": "nonfiction_essay",
    "political commentary": "nonfiction_essay",
    "social commentary": "nonfiction_essay",
    "nature writing": "nonfiction_essay",
    "religious text": "religious_philosophical",
    "religious literature": "religious_philosophical",
    "religious fiction": "religious_philosophical",
    "philosophical essay": "religious_philosophical",
    "philosophical fiction": "religious_philosophical",
    "philosophical": "religious_philosophical",
    "religious": "religious_philosophical",
}

SPLIT_RE = re.compile(r"\s*(?:/|,|;|\+|\band\b|&)\s*")


def normalize_genre(raw: str | None) -> dict[str, Any]:
    """Map ``genre_raw`` into a compact normalized genre label."""
    if raw is None or not str(raw).strip():
        return _result(raw, [], [], "other", False, True, "missing genre_raw")

    candidates = parse_genre_candidates(str(raw))
    mapped = [_map_candidate(candidate) for candidate in candidates]
    mapped_labels = [label for label in mapped if label is not None]
    unique_labels = _unique(mapped_labels)

    if not unique_labels:
        return _result(
            raw,
            candidates,
            [],
            "other",
            False,
            True,
            "no top-label rule matched",
        )

    primary = _choose_primary_label(candidates, mapped)
    if len(unique_labels) == 1:
        return _result(
            raw,
            candidates,
            unique_labels,
            unique_labels[0],
            True,
            False,
            "single normalized genre",
        )

    return _result(
        raw,
        candidates,
        unique_labels,
        primary,
        False,
        True,
        "compound genre maps to multiple normalized labels",
    )


def parse_genre_candidates(raw: str) -> list[str]:
    cleaned = clean_value(raw)
    parts = [part.strip() for part in SPLIT_RE.split(cleaned) if part.strip()]
    return parts or ([cleaned] if cleaned else [])


def clean_value(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9/&,+; ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_records(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for record in records:
        normalized = normalize_genre(record.get("genre_raw"))
        next_record = dict(record)
        next_record.update(normalized)
        yield next_record


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
            writer.writerow({"text": record.get("full_text") or "", "label": record["genre_norm"]})
            count += 1
    return count


def write_frequencies(counter: Counter[str], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["genre_raw", "count"])
        for value, count in counter.most_common():
            writer.writerow([value, count])


def write_uncertain(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "instruction_id",
            "genre_raw",
            "genre_candidates",
            "genre_candidate_labels",
            "genre_norm",
            "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if record["genre_uncertain"] or not record["genre_keep"]:
                writer.writerow(
                    {
                        "instruction_id": record.get("instruction_id"),
                        "genre_raw": record.get("genre_raw"),
                        "genre_candidates": json.dumps(
                            record.get("genre_candidates", []), ensure_ascii=False
                        ),
                        "genre_candidate_labels": json.dumps(
                            record.get("genre_candidate_labels", []), ensure_ascii=False
                        ),
                        "genre_norm": record.get("genre_norm"),
                        "reason": record.get("genre_norm_reason"),
                    }
                )
                count += 1
    return count


def is_clean_classification_record(record: dict[str, Any]) -> bool:
    return (
        record.get("genre_keep") is True
        and record.get("genre_norm") != "other"
        and not record.get("genre_uncertain", True)
        and bool(record.get("full_text"))
    )


def _map_candidate(candidate: str) -> str | None:
    direct = DIRECT_ALIASES.get(candidate)
    if direct:
        return direct

    if "historical" in candidate:
        return "historical"
    if any(term in candidate for term in ("drama", "traged", "play")):
        return "drama"
    if "romance" in candidate or "romantic" in candidate:
        return "romance"
    if any(term in candidate for term in ("adventure", "western")):
        return "adventure"
    if "poetry" in candidate or "poem" in candidate:
        return "poetry"
    if any(term in candidate for term in ("children", "young adult", "coming of age")):
        return "children_young_adult"
    if any(term in candidate for term in ("mystery", "crime", "detective", "thriller")):
        return "mystery_crime"
    if any(term in candidate for term in ("comedy", "satire", "humor")):
        return "comedy_satire"
    if any(term in candidate for term in ("science fiction", "fantasy", "gothic", "horror")):
        return "speculative"
    if any(
        term in candidate
        for term in (
            "essay",
            "non fiction",
            "biography",
            "memoir",
            "travel",
            "criticism",
            "commentary",
            "discourse",
            "nature writing",
        )
    ):
        return "nonfiction_essay"
    if "religious" in candidate or "philosophical" in candidate:
        return "religious_philosophical"
    return None


def _choose_primary_label(candidates: list[str], mapped: list[str | None]) -> str:
    for label in mapped:
        if label is not None:
            return label
    labels = [label for label in mapped if label is not None]
    return min(labels, key=LABEL_PRIORITY.index) if labels else "other"


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _result(
    raw: str | None,
    candidates: list[str],
    candidate_labels: list[str],
    label: str,
    keep: bool,
    uncertain: bool,
    reason: str,
) -> dict[str, Any]:
    if label not in LABELS:
        raise ValueError(f"unknown normalized genre label: {label}")
    return {
        "genre_raw": raw,
        "genre_candidates": candidates,
        "genre_candidate_labels": candidate_labels,
        "genre_norm": label,
        "genre_keep": keep,
        "genre_uncertain": uncertain,
        "genre_norm_reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/brahe_novels_parsed.jsonl")
    parser.add_argument("--output", default="data/brahe_novels_genre_normalized.jsonl")
    parser.add_argument("--classification-output", default="data/brahe_genre_classification.csv")
    parser.add_argument("--frequency-output", default="data/genre_raw_frequencies.csv")
    parser.add_argument("--uncertain-output", default="data/genre_uncertain_cases.csv")
    parser.add_argument("--examples", type=int, default=16)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    frequencies = Counter(row.get("genre_raw") or "" for row in rows)
    normalized_rows = list(normalize_records(rows))
    label_counts = Counter(row["genre_norm"] for row in normalized_rows)
    keep_counts = Counter("keep" if row["genre_keep"] else "drop" for row in normalized_rows)

    write_frequencies(frequencies, args.frequency_output)
    normalized_count = write_jsonl(normalized_rows, args.output)
    classification_count = write_classification_csv(normalized_rows, args.classification_output)
    uncertain_count = write_uncertain(normalized_rows, args.uncertain_output)

    print(f"Wrote {normalized_count} normalized records to {args.output}")
    print(f"Wrote {classification_count} classification rows to {args.classification_output}")
    print(f"Wrote raw value frequencies to {args.frequency_output}")
    print(f"Wrote {uncertain_count} uncertain/drop cases to {args.uncertain_output}")
    print("Most common raw genre values:")
    for value, count in frequencies.most_common(25):
        print(f"{count}\t{value!r}")
    print("Normalized genre counts:")
    for label, count in label_counts.most_common():
        print(f"{label}\t{count}")
    print("Keep/drop counts:")
    for status, count in keep_counts.most_common():
        print(f"{status}\t{count}")
    print("Before/after examples:")
    for row in normalized_rows[: args.examples]:
        print(
            json.dumps(
                {
                    "genre_raw": row.get("genre_raw"),
                    "genre_candidates": row.get("genre_candidates"),
                    "genre_norm": row.get("genre_norm"),
                    "genre_keep": row.get("genre_keep"),
                    "genre_uncertain": row.get("genre_uncertain"),
                    "reason": row.get("genre_norm_reason"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
