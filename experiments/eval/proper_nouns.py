#!/usr/bin/env python3
"""
Proper noun accuracy evaluation.

Extracts entity-tagged proper nouns from Earnings-21's NLP reference files
and measures whether they appear in scribe's hypothesis text.

This complements WER, which averages all words equally. Proper nouns
matter disproportionately for downstream LLM processing — getting "Claude"
or "Anthropic" wrong breaks grounding even at otherwise good WER.
"""

import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

# Entity types we consider "proper nouns" for the recall metric
PROPER_NOUN_TYPES = {
    "PERSON",       # Frank Lopez-Balboa
    "ORG",          # Cumulus Media, GAAP, EBITDA
    "GPE",          # Nashville (geo-political entity)
    "PRODUCT",      # branded products
    "LAW",          # CARES Act
    "WORK_OF_ART",  # named shows / publications
    "FAC",          # named facilities
}

CACHE_DIR = Path(__file__).parent / "datasets" / "earnings21" / "_proper_nouns_cache"


def extract_proper_nouns_earnings21(file_id: str) -> list[tuple[str, str]]:
    """
    Extract list of (phrase, entity_type) tuples from an Earnings-21 file's NLP reference.

    Consecutive tokens with the same wer_tag id are merged into one phrase
    (e.g. "Frank Lopez-Balboa" rather than separate tokens).
    """
    nlp_path = hf_hub_download(
        "Revai/earnings21", f"nlp_references/{file_id}.nlp", repo_type="dataset"
    )
    tag_path = hf_hub_download(
        "Revai/earnings21", f"wer_tags/{file_id}.wer_tag.json", repo_type="dataset"
    )

    with open(tag_path) as f:
        tag_lookup = json.load(f)

    phrases = []
    current_phrase = []
    current_type = None
    current_tag_id = None

    with open(nlp_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 8:
                continue
            token = parts[0]
            wer_tags = [
                t.strip().strip("'") for t in parts[7].strip("[]").split(",") if t.strip()
            ]

            token_type = None
            token_tag_id = None
            for tid in wer_tags:
                if tid in tag_lookup:
                    et = tag_lookup[tid].get("entity_type")
                    if et in PROPER_NOUN_TYPES:
                        token_type = et
                        token_tag_id = tid
                        break

            if token_type and token_tag_id == current_tag_id:
                current_phrase.append(token)
            else:
                if current_phrase:
                    phrases.append((" ".join(current_phrase), current_type))
                if token_type:
                    current_phrase = [token]
                    current_type = token_type
                    current_tag_id = token_tag_id
                else:
                    current_phrase = []
                    current_type = None
                    current_tag_id = None

    if current_phrase:
        phrases.append((" ".join(current_phrase), current_type))

    return phrases


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def score_recall(phrases: list[tuple[str, str]], hypothesis: str) -> dict:
    """
    Compute proper noun recall: % of unique expected phrases that appear in hypothesis.

    Uses word-boundary substring matching after normalization (lowercase, no punct).
    Returns dict with: total, found, missed, recall, missed_list, found_list.
    """
    norm_hyp = normalize(hypothesis)

    # Deduplicate phrases by normalized form
    unique = {}
    for phrase, etype in phrases:
        key = normalize(phrase)
        if key and key not in unique:
            unique[key] = (phrase, etype)

    found = []
    missed = []
    for norm_phrase, (orig_phrase, etype) in unique.items():
        pattern = r"\b" + re.escape(norm_phrase) + r"\b"
        if re.search(pattern, norm_hyp):
            found.append((orig_phrase, etype))
        else:
            missed.append((orig_phrase, etype))

    total = len(unique)
    return {
        "total": total,
        "found": len(found),
        "missed": len(missed),
        "recall": len(found) / total if total else 0.0,
        "missed_list": missed,
        "found_list": found,
    }


def get_proper_nouns(file_id: str) -> list[tuple[str, str]]:
    """Get cached proper nouns for an Earnings-21 file (downloads on first use)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{file_id}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return [tuple(p) for p in json.load(f)]

    phrases = extract_proper_nouns_earnings21(file_id)
    with open(cache_path, "w") as f:
        json.dump(phrases, f)
    return phrases


if __name__ == "__main__":
    # Self-test on Cumulus Media
    phrases = get_proper_nouns("4386541")
    print(f"Extracted {len(phrases)} proper noun mentions")
    unique = set(normalize(p[0]) for p in phrases)
    print(f"Unique entities: {len(unique)}")
