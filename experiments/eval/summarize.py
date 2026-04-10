#!/usr/bin/env python3
"""
Deterministic summary of all eval results from CSV files.

Reads all result CSVs and produces a Markdown-formatted summary table.
This script is the single source of truth for accuracy claims.

Usage:
    python3 summarize.py                    # Print summary to stdout
    python3 summarize.py --format markdown  # Markdown table (default)
    python3 summarize.py --format csv       # CSV output
"""

import argparse
import csv
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_all_results() -> list[dict]:
    """Load all CSV result files."""
    results = []
    for csv_path in sorted(RESULTS_DIR.glob("scribe-*.csv")):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source"] = csv_path.name
                results.append(row)
    return results


def aggregate_by_group(results: list[dict], group_key: str) -> dict:
    """Aggregate WER by a grouping key (dataset, model, etc.)."""
    groups = {}
    for r in results:
        if r.get("error"):
            continue
        key = r[group_key]
        if key not in groups:
            groups[key] = {
                "files": 0,
                "total_ref": 0,
                "total_sub": 0,
                "total_del": 0,
                "total_ins": 0,
                "total_dur": 0,
                "total_proc": 0,
            }
        g = groups[key]
        g["files"] += 1
        g["total_ref"] += int(r["ref_words"])
        g["total_sub"] += int(r["substitutions"])
        g["total_del"] += int(r["deletions"])
        g["total_ins"] += int(r["insertions"])
        g["total_dur"] += float(r["duration_secs"])
        g["total_proc"] += float(r["processing_secs"])
    return groups


def compute_wer(g: dict) -> float:
    if g["total_ref"] == 0:
        return 0.0
    return (g["total_sub"] + g["total_del"] + g["total_ins"]) / g["total_ref"]


def compute_rtf(g: dict) -> float:
    if g["total_dur"] == 0:
        return 0.0
    return g["total_proc"] / g["total_dur"]


def print_markdown(results: list[dict]):
    # Filter out dipco (incompatible dataset)
    results = [r for r in results if r.get("dataset") != "dipco" and not r.get("error")]

    # --- Table 1: Dataset × Model matrix ---
    print("## Accuracy by Dataset and Model\n")
    print("Aggregate WER (Word Error Rate) — lower is better.\n")

    # Get unique datasets and models
    combos = {}
    for r in results:
        key = (r["dataset"], r["model"])
        if key not in combos:
            combos[key] = {
                "total_ref": 0, "total_sub": 0, "total_del": 0,
                "total_ins": 0, "total_dur": 0, "files": 0,
            }
        c = combos[key]
        c["files"] += 1
        c["total_ref"] += int(r["ref_words"])
        c["total_sub"] += int(r["substitutions"])
        c["total_del"] += int(r["deletions"])
        c["total_ins"] += int(r["insertions"])
        c["total_dur"] += float(r["duration_secs"])

    datasets = sorted(set(r["dataset"] for r in results))
    models = sorted(set(r["model"] for r in results))

    # Dataset display names
    ds_names = {
        "earnings21": "Earnings-21 (multi-speaker, phone audio)",
        "tedlium": "TED-LIUM 3 (single-speaker, professional audio)",
        "mls-spanish": "MLS Spanish (single-speaker, audiobook)",
    }

    print("| Dataset | " + " | ".join(models) + " |")
    print("|" + "---|" * (len(models) + 1))

    for ds in datasets:
        row = f"| {ds_names.get(ds, ds)} |"
        for model in models:
            key = (ds, model)
            if key in combos:
                c = combos[key]
                wer = compute_wer(c)
                row += f" **{wer:.1%}** ({c['files']} files, {c['total_dur']/60:.0f}m) |"
            else:
                row += " — |"
        print(row)

    # --- Proper noun recall section (Earnings-21 only, split by mode) ---
    pn_results = [r for r in results if r["dataset"] == "earnings21" and int(r.get("proper_noun_total", 0)) > 0]
    if pn_results:
        print("\n## Proper Noun Recall (Earnings-21)\n")
        print("WER averages all words equally. Proper noun recall measures how many entity-tagged ")
        print("phrases (PERSON, ORG, GPE, PRODUCT, LAW, WORK_OF_ART, FAC) appear in the hypothesis.\n")

        modes = sorted(set(r.get("mode", "batch") for r in pn_results))
        for mode in modes:
            mode_results = [r for r in pn_results if r.get("mode", "batch") == mode]
            if not mode_results:
                continue
            label_model = mode_results[0]["model"]
            print(f"\n### {mode} ({label_model})\n")
            print("| File | Company | WER | Proper Noun Recall |")
            print("|---|---|---|---|")
            total_pn = found_pn = 0
            for r in sorted(mode_results, key=lambda x: x["file_id"]):
                wer = float(r["wer"])
                pn_total = int(r["proper_noun_total"])
                pn_found = int(r["proper_noun_found"])
                pn_recall = float(r["proper_noun_recall"])
                print(f"| {r['file_id']} | {r['label']} | {wer:.1%} | {pn_recall:.1%} ({pn_found}/{pn_total}) |")
                total_pn += pn_total
                found_pn += pn_found
            agg_recall = found_pn / total_pn if total_pn else 0
            print(f"\n**Aggregate**: {agg_recall:.1%} ({found_pn}/{total_pn} entities)")

    # --- Table 2: Per-file detail for primary model ---
    primary_model = "large-v3-turbo"
    primary = [r for r in results if r["model"] == primary_model]

    print(f"\n## Per-file Results ({primary_model})\n")
    print("| Dataset | File | Duration | WER | Substitutions | Deletions | Insertions | Ref Words |")
    print("|---|---|---|---|---|---|---|---|")

    for r in sorted(primary, key=lambda x: (x["dataset"], x["file_id"])):
        dur_min = float(r["duration_secs"]) / 60
        print(f"| {r['dataset']} | {r['file_id']} | {dur_min:.1f}m | {float(r['wer']):.1%} | "
              f"{r['substitutions']} | {r['deletions']} | {r['insertions']} | {r['ref_words']} |")

    # --- Summary stats ---
    print(f"\n## Summary\n")
    turbo_all = [r for r in results if r["model"] == primary_model]
    total_ref = sum(int(r["ref_words"]) for r in turbo_all)
    total_errors = sum(int(r["substitutions"]) + int(r["deletions"]) + int(r["insertions"]) for r in turbo_all)
    total_dur = sum(float(r["duration_secs"]) for r in turbo_all)
    total_proc = sum(float(r["processing_secs"]) for r in turbo_all)

    # Detect actual ASR engine based on scribe version
    versions = sorted(set(r.get("scribe_version", "?") for r in turbo_all))
    version_str = ", ".join(versions)
    if any(v.startswith("0.2") or v.startswith("0.3") for v in versions):
        engine = "NVIDIA Parakeet TDT v3 via FluidAudio (CoreML, Apple Silicon)"
    else:
        engine = f"Whisper {primary_model} via WhisperKit (CoreML, Apple Silicon)"
    print(f"- **scribe version**: {version_str}")
    print(f"- **ASR engine**: {engine}")
    print(f"- **Total audio evaluated**: {total_dur/60:.0f} minutes across {len(turbo_all)} files")
    print(f"- **Overall WER**: {total_errors/total_ref:.1%}")
    print(f"- **Average RTF**: {total_proc/total_dur:.3f}x ({total_dur/total_proc:.0f}x faster than real-time)")
    print(f"- **Evaluation tool**: jiwer (deterministic WER computation)")
    print(f"- **Text normalization**: lowercase, remove punctuation, collapse whitespace")
    print(f"- **Reproducibility**: same audio + same model = identical WER (only wall-clock time varies)")


def main():
    parser = argparse.ArgumentParser(description="Summarize eval results")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    args = parser.parse_args()

    results = load_all_results()
    if not results:
        print("No result files found.", file=sys.stderr)
        return

    if args.format == "markdown":
        print_markdown(results)


if __name__ == "__main__":
    main()
