#!/usr/bin/env python3
"""
Deterministic evaluation of scribe transcription quality against ground-truth datasets.

Supports multiple datasets (Earnings-21, DiPCo) and Whisper model variants.

Usage:
    python3 eval.py                                  # All datasets, default model
    python3 eval.py --dataset earnings21             # Earnings-21 only
    python3 eval.py --dataset dipco                  # DiPCo only
    python3 eval.py --model large-v3                 # Use large-v3 (most accurate)
    python3 eval.py --model large-v3-turbo           # Use large-v3-turbo (fastest)
    python3 eval.py --dataset earnings21 --model large-v3 --max-files 2
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

import jiwer
from huggingface_hub import hf_hub_download

from proper_nouns import get_proper_nouns, score_recall

# --- Configuration ---

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"

# Text normalization for fair WER comparison
# Must end with ReduceToListOfListOfWords for jiwer 4.x
JIWER_TRANSFORMS = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

# --- Earnings-21 dataset ---

EARNINGS21_ID = "Revai/earnings21"
EARNINGS21_DIR = EVAL_DIR / "datasets" / "earnings21"
EARNINGS21_FILES = [
    "4386541",  # 18.3m - Cumulus Media (5 speakers)
    "4387332",  # 21.8m - Zagg Inc (6 speakers)
    "4392809",  # 24.0m - Lands End (6 speakers)
    "4384683",  # 24.6m - One Gas (8 speakers)
    "4367318",  # 28.7m - AcelRX Pharmaceuticals (7 speakers)
]

# --- TED-LIUM dataset ---

TEDLIUM_DIR = EVAL_DIR / "datasets" / "tedlium"
TEDLIUM_FILES = ["talk_0", "talk_1", "talk_2", "talk_3", "talk_4"]

# --- MLS Spanish dataset ---

MLS_SPANISH_DIR = EVAL_DIR / "datasets" / "mls-spanish"

# --- DiPCo dataset (verbatim — incompatible with Whisper WER eval) ---

DIPCO_DIR = EVAL_DIR / "datasets" / "dipco" / "Dipco"
DIPCO_SESSIONS = ["S08", "S07", "S01"]


# ========== Dataset loaders ==========

def load_earnings21(file_ids: list[str], download: bool = True) -> list[dict]:
    """Load Earnings-21 dataset entries."""
    EARNINGS21_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = hf_hub_download(EARNINGS21_ID, "metadata.jsonl", repo_type="dataset")
    with open(meta_path) as f:
        all_meta = {
            json.loads(line)["file_name"].split("/")[1].replace(".wav", ""): json.loads(line)
            for line in f
        }

    entries = []
    for fid in file_ids:
        if fid not in all_meta:
            print(f"  WARNING: {fid} not found in metadata, skipping")
            continue

        meta = all_meta[fid]
        wav_path = EARNINGS21_DIR / f"{fid}.wav"
        ref_path = EARNINGS21_DIR / f"{fid}.ref.txt"

        if not wav_path.exists() and download:
            print(f"  Downloading {fid}.wav ({meta['audio_length']/60:.1f}m)...")
            downloaded = hf_hub_download(EARNINGS21_ID, f"wav/{fid}.wav", repo_type="dataset")
            os.symlink(os.path.abspath(downloaded), wav_path)
        elif not wav_path.exists():
            print(f"  WARNING: {fid}.wav not found, skipping")
            continue

        if not ref_path.exists():
            ref_path.write_text(meta["text"])

        entries.append({
            "file_id": fid,
            "dataset": "earnings21",
            "label": meta.get("company_name", fid),
            "wav_path": wav_path,
            "ref_text": ref_path.read_text(),
            "duration_secs": meta["audio_length"],
            "speakers": meta.get("unique_speakers", 0),
        })

    return entries


def load_tedlium(file_ids: list[str]) -> list[dict]:
    """Load TED-LIUM long-form test talks (pre-downloaded)."""
    entries = []
    for fid in file_ids:
        wav_path = TEDLIUM_DIR / f"{fid}.wav"
        ref_path = TEDLIUM_DIR / f"{fid}.ref.txt"

        if not wav_path.exists():
            print(f"  WARNING: {fid} not found — run download_datasets.py first")
            continue

        import soundfile
        info = soundfile.info(str(wav_path))

        entries.append({
            "file_id": fid,
            "dataset": "tedlium",
            "label": f"TED talk {fid}",
            "wav_path": wav_path,
            "ref_text": ref_path.read_text(),
            "duration_secs": info.duration,
            "speakers": 1,
        })
    return entries


def load_mls_spanish(max_files: int = 5) -> list[dict]:
    """Load MLS Spanish test chapters (pre-downloaded)."""
    import soundfile

    entries = []
    wav_files = sorted(MLS_SPANISH_DIR.glob("chapter_*.wav"))

    for wav_path in wav_files[:max_files]:
        fid = wav_path.stem
        ref_path = wav_path.with_suffix(".ref.txt")

        if not ref_path.exists():
            print(f"  WARNING: {fid} ref not found, skipping")
            continue

        info = soundfile.info(str(wav_path))

        entries.append({
            "file_id": fid,
            "dataset": "mls-spanish",
            "label": f"MLS Spanish {fid}",
            "wav_path": wav_path,
            "ref_text": ref_path.read_text(),
            "duration_secs": info.duration,
            "speakers": 1,
        })
    return entries


def load_dipco(session_ids: list[str]) -> list[dict]:
    """Load DiPCo dataset entries (far-field single mic)."""
    entries = []

    for sid in session_ids:
        # Prefer mixed close-talk mics (realistic), fall back to far-field
        wav_path = DIPCO_DIR / "audio" / "eval" / f"{sid}_mixed.wav"
        trans_path = DIPCO_DIR / "transcriptions" / "eval" / f"{sid}.json"

        if not wav_path.exists():
            wav_path = DIPCO_DIR / "audio" / "eval" / f"{sid}_U01.CH1.wav"
        if not wav_path.exists():
            wav_path = DIPCO_DIR / "audio" / "dev" / f"{sid}_mixed.wav"
            trans_path = DIPCO_DIR / "transcriptions" / "dev" / f"{sid}.json"

        if not wav_path.exists():
            print(f"  WARNING: {sid} audio not found, skipping")
            continue
        if not trans_path.exists():
            print(f"  WARNING: {sid} transcription not found, skipping")
            continue

        # Parse transcription JSON — list of utterances with "words" field
        with open(trans_path) as f:
            utterances = json.load(f)

        # Concatenate all utterance words in time order for reference text
        # Filter out noise/non-speech tags
        ref_parts = []
        for utt in utterances:
            words = utt.get("words", "").strip()
            # Skip noise-only utterances
            if not words or words in ("[noise]", "[unintelligible]", "[laughter]"):
                continue
            # Remove inline tags but keep the surrounding words
            words = re.sub(r"\[(?:noise|unintelligible|laughter|overlap)\]", "", words)
            words = words.strip()
            if words:
                ref_parts.append(words)

        ref_text = " ".join(ref_parts)
        speakers = len(set(u.get("speaker_id", "") for u in utterances))

        # Get duration from audio file
        import soundfile
        info = soundfile.info(str(wav_path))
        duration = info.duration

        entries.append({
            "file_id": sid,
            "dataset": "dipco",
            "label": f"DiPCo {sid} (dinner party)",
            "wav_path": wav_path,
            "ref_text": ref_text,
            "duration_secs": duration,
            "speakers": speakers,
        })

    return entries


# ========== Core eval functions ==========

def run_scribe(wav_path: Path, model: str = "large-v3-turbo",
               scribe_bin: str = "scribe", mode: str = "batch") -> tuple[str, float]:
    """Run scribe on an audio file. Returns (transcript, elapsed_secs).

    mode="batch":     calls `scribe transcribe ...` (offline, full audio at once)
    mode="streaming": calls `scribe stream --audio-file ...` (chunked through streaming engine)
    """
    if mode == "streaming":
        return _run_scribe_streaming(wav_path, scribe_bin)
    return _run_scribe_batch(wav_path, model, scribe_bin)


def _run_scribe_batch(wav_path: Path, model: str, scribe_bin: str) -> tuple[str, float]:
    cmd = [scribe_bin, "transcribe", str(wav_path), "--format", "txt"]

    # Check if scribe supports --model flag (v0.1.x only — v0.2+ uses Parakeet by default)
    version_result = subprocess.run([scribe_bin, "--version"], capture_output=True, text=True)
    version = version_result.stdout.strip()
    if version.startswith("0.1"):
        cmd.extend(["--model", model])

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        raise RuntimeError(f"scribe failed on {wav_path}: {result.stderr}")

    # Strip timestamp/speaker prefixes from scribe txt output
    lines = result.stdout.strip().split("\n")
    text_parts = []
    for line in lines:
        cleaned = re.sub(r"^\[[\d:]+\]\s*(?:(?:Speaker\s+\d+|Unknown):\s*)?", "", line)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = cleaned.strip()
        if cleaned and cleaned != "-":
            text_parts.append(cleaned)

    return " ".join(text_parts), elapsed


def _run_scribe_streaming(wav_path: Path, scribe_bin: str) -> tuple[str, float]:
    """Feed an audio file through `scribe stream --audio-file` and parse JSONL output.

    Each JSONL line is `{"text": "...", "time": 1.2}` representing a confirmed delta.
    Concatenate the `text` fields in order to build the full hypothesis.
    """
    cmd = [
        scribe_bin, "stream",
        "--audio-file", str(wav_path),
        "--engine", "nemotron",
        "--format", "jsonl",
    ]

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        raise RuntimeError(f"scribe stream failed on {wav_path}: {result.stderr}")

    text_parts = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get("text", "").strip()
        if text:
            text_parts.append(text)

    return " ".join(text_parts), elapsed


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute WER and related metrics."""
    output = jiwer.process_words(
        reference,
        hypothesis,
        reference_transform=JIWER_TRANSFORMS,
        hypothesis_transform=JIWER_TRANSFORMS,
    )
    return {
        "wer": output.wer,
        "mer": output.mer,
        "wil": output.wil,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "ref_words": sum(len(ref) for ref in output.references),
        "hyp_words": sum(len(hyp) for hyp in output.hypotheses),
    }


def get_scribe_version(scribe_bin: str = "scribe") -> str:
    result = subprocess.run([scribe_bin, "--version"], capture_output=True, text=True)
    return result.stdout.strip()


def run_eval(entries: list[dict], model: str, scribe_version: str,
             scribe_bin: str = "scribe", mode: str = "batch") -> list[dict]:
    """Run evaluation on all entries."""
    results = []

    for entry in entries:
        fid = entry["file_id"]
        label = entry["label"]
        dur_min = entry["duration_secs"] / 60

        print(f"\n--- {fid}: {label} ({dur_min:.1f}m, {entry['speakers']} speakers) ---")
        if mode == "streaming":
            print(f"  Streaming via scribe stream --audio-file (engine: nemotron)...")
        else:
            print(f"  Transcribing with {model}...")

        try:
            hypothesis, elapsed = run_scribe(entry["wav_path"], model=model, scribe_bin=scribe_bin, mode=mode)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "file_id": fid,
                "dataset": entry["dataset"],
                "label": label,
                "model": "nemotron-streaming" if mode == "streaming" else model,
                "mode": mode,
                "duration_secs": round(entry["duration_secs"], 1),
                "speakers": entry["speakers"],
                "scribe_version": scribe_version,
                "processing_secs": 0,
                "rtf": 0,
                "wer": 0, "mer": 0, "wil": 0,
                "substitutions": 0, "deletions": 0, "insertions": 0,
                "ref_words": 0, "hyp_words": 0,
                "proper_noun_total": 0, "proper_noun_found": 0, "proper_noun_recall": 0,
                "error": str(e),
            })
            continue

        print("  Computing WER...")
        metrics = compute_wer(entry["ref_text"], hypothesis)
        rtf = elapsed / entry["duration_secs"]

        # Proper noun recall (only for Earnings-21 — has built-in entity tags)
        pn_total = 0
        pn_found = 0
        pn_recall = 0.0
        if entry["dataset"] == "earnings21":
            try:
                phrases = get_proper_nouns(entry["file_id"])
                pn_result = score_recall(phrases, hypothesis)
                pn_total = pn_result["total"]
                pn_found = pn_result["found"]
                pn_recall = pn_result["recall"]
                print(f"  WER: {metrics['wer']:.1%} | Proper noun recall: {pn_recall:.1%} ({pn_found}/{pn_total}) | "
                      f"RTF: {rtf:.3f}x | S/D/I: {metrics['substitutions']}/{metrics['deletions']}/{metrics['insertions']}")
            except Exception as e:
                print(f"  WARNING: proper noun scoring failed: {e}")
                print(f"  WER: {metrics['wer']:.1%} | RTF: {rtf:.3f}x | "
                      f"S/D/I: {metrics['substitutions']}/{metrics['deletions']}/{metrics['insertions']}")
        else:
            print(f"  WER: {metrics['wer']:.1%} | RTF: {rtf:.3f}x | "
                  f"S/D/I: {metrics['substitutions']}/{metrics['deletions']}/{metrics['insertions']}")

        results.append({
            "file_id": fid,
            "dataset": entry["dataset"],
            "label": label,
            "model": "nemotron-streaming" if mode == "streaming" else model,
            "mode": mode,
            "duration_secs": round(entry["duration_secs"], 1),
            "speakers": entry["speakers"],
            "scribe_version": scribe_version,
            "processing_secs": round(elapsed, 1),
            "rtf": round(rtf, 4),
            "wer": round(metrics["wer"], 4),
            "mer": round(metrics["mer"], 4),
            "wil": round(metrics["wil"], 4),
            "substitutions": metrics["substitutions"],
            "deletions": metrics["deletions"],
            "insertions": metrics["insertions"],
            "ref_words": metrics["ref_words"],
            "hyp_words": metrics["hyp_words"],
            "proper_noun_total": pn_total,
            "proper_noun_found": pn_found,
            "proper_noun_recall": round(pn_recall, 4),
            "error": "",
        })

    return results


def write_csv(results: list[dict], output_path: Path):
    """Write evaluation results to CSV."""
    if not results:
        print("No results to write.")
        return

    fieldnames = [
        "file_id", "dataset", "label", "model", "mode",
        "duration_secs", "speakers", "scribe_version",
        "processing_secs", "rtf",
        "wer", "mer", "wil",
        "substitutions", "deletions", "insertions",
        "ref_words", "hyp_words",
        "proper_noun_total", "proper_noun_found", "proper_noun_recall",
        "error",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}")


def print_summary(results: list[dict]):
    """Print summary statistics grouped by dataset."""
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print("No valid results.")
        return

    # Group by dataset
    datasets = sorted(set(r["dataset"] for r in valid))

    print("\n" + "=" * 70)
    print(f"SUMMARY — model: {valid[0]['model']}")
    print("=" * 70)

    for ds in datasets:
        ds_results = [r for r in valid if r["dataset"] == ds]
        total_ref = sum(r["ref_words"] for r in ds_results)
        total_sub = sum(r["substitutions"] for r in ds_results)
        total_del = sum(r["deletions"] for r in ds_results)
        total_ins = sum(r["insertions"] for r in ds_results)
        total_dur = sum(r["duration_secs"] for r in ds_results)
        total_proc = sum(r["processing_secs"] for r in ds_results)
        agg_wer = (total_sub + total_del + total_ins) / total_ref if total_ref else 0

        print(f"\n  {ds} ({len(ds_results)} files, {total_dur/60:.1f}m audio)")
        print(f"    Aggregate WER:  {agg_wer:.1%}")
        print(f"    Average RTF:    {total_proc/total_dur:.4f}x")
        print(f"    WER range:      {min(r['wer'] for r in ds_results):.1%} - {max(r['wer'] for r in ds_results):.1%}")
        print(f"    S/D/I:          {total_sub}/{total_del}/{total_ins} (of {total_ref} ref words)")

    # Overall
    total_ref = sum(r["ref_words"] for r in valid)
    total_errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in valid)
    if total_ref > 0:
        print(f"\n  OVERALL: {total_errors/total_ref:.1%} WER across {len(valid)} files")
    else:
        print(f"\n  OVERALL: N/A (0 reference words) across {len(valid)} files")


def main():
    parser = argparse.ArgumentParser(description="Evaluate scribe transcription quality")
    parser.add_argument("--dataset", choices=["earnings21", "tedlium", "mls-spanish", "dipco", "all"], default="all",
                        help="Dataset to evaluate (default: all)")
    parser.add_argument("--model", default="large-v3-turbo",
                        help="Whisper model (default: large-v3-turbo)")
    parser.add_argument("--max-files", type=int, default=5,
                        help="Max files per dataset (default: 5)")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--output", type=str, help="Output CSV path")
    parser.add_argument("--scribe-bin", default="scribe", help="Path to scribe binary")
    parser.add_argument("--mode", choices=["batch", "streaming"], default="batch",
                        help="Eval mode: 'batch' (scribe transcribe) or 'streaming' (scribe stream --audio-file)")
    args = parser.parse_args()

    scribe_version = get_scribe_version(args.scribe_bin)
    print(f"scribe version: {scribe_version} ({args.scribe_bin})")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")

    # Load datasets
    entries = []

    if args.dataset in ("earnings21", "all"):
        print("\n--- Loading Earnings-21 ---")
        file_ids = EARNINGS21_FILES[:args.max_files]
        entries.extend(load_earnings21(file_ids))

    if args.dataset in ("tedlium", "all"):
        print("\n--- Loading TED-LIUM ---")
        file_ids = TEDLIUM_FILES[:args.max_files]
        entries.extend(load_tedlium(file_ids))

    if args.dataset in ("mls-spanish", "all"):
        if args.mode == "streaming":
            print("\n--- Skipping MLS Spanish (Nemotron streaming engine is English-only) ---")
        else:
            print("\n--- Loading MLS Spanish ---")
            entries.extend(load_mls_spanish(args.max_files))

    if args.dataset == "dipco":
        print("\n--- Loading DiPCo ---")
        session_ids = DIPCO_SESSIONS[:args.max_files]
        entries.extend(load_dipco(session_ids))

    if args.download_only:
        print("Download complete.")
        return

    if not entries:
        print("ERROR: No dataset entries loaded.")
        return

    # Run eval
    print(f"\n--- Running evaluation ({len(entries)} files) ---")
    results = run_eval(entries, args.model, scribe_version, scribe_bin=args.scribe_bin, mode=args.mode)

    # Write results
    if args.output:
        output_path = Path(args.output)
    elif args.mode == "streaming":
        output_path = RESULTS_DIR / f"scribe-{scribe_version}-streaming-{args.dataset}.csv"
    else:
        output_path = RESULTS_DIR / f"scribe-{scribe_version}-{args.model}-{args.dataset}.csv"
    write_csv(results, output_path)
    print_summary(results)


if __name__ == "__main__":
    main()
