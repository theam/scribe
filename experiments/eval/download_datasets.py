#!/usr/bin/env python3
"""
Download and prepare evaluation datasets for scribe.

Prepares audio files as WAV with matching reference text files.

Usage:
    python3 download_datasets.py                  # Download all
    python3 download_datasets.py --dataset tedlium # Download specific dataset
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf

DATASETS_DIR = Path(__file__).parent / "datasets"


def download_tedlium(max_files: int = 5):
    """Download TED-LIUM 3 long-form test talks."""
    from datasets import load_dataset

    out_dir = DATASETS_DIR / "tedlium"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TED-LIUM long-form test split (streaming)...")
    ds = load_dataset("distil-whisper/tedlium-long-form", split="test", streaming=True)

    count = 0
    for i, sample in enumerate(ds):
        audio = sample["audio"]
        dur = len(audio["array"]) / audio["sampling_rate"]

        # Skip very short samples (<60s)
        if dur < 60:
            print(f"  Skipping sample {i} ({dur:.0f}s — too short)")
            continue

        fid = f"talk_{count}"
        wav_path = out_dir / f"{fid}.wav"
        ref_path = out_dir / f"{fid}.ref.txt"
        meta_path = out_dir / f"{fid}.meta.txt"

        if wav_path.exists():
            print(f"  {fid} already exists, skipping")
            count += 1
            if count >= max_files:
                break
            continue

        print(f"  Saving {fid}: {dur:.0f}s ({dur/60:.1f}m)")

        # Save WAV
        array = np.array(audio["array"], dtype=np.float32)
        sf.write(str(wav_path), array, audio["sampling_rate"])

        # Save reference text
        ref_path.write_text(sample.get("text", ""))

        # Save metadata
        meta_path.write_text(f"duration_secs: {dur:.1f}\nsource: TED-LIUM 3\nid: {sample.get('id', i)}\n")

        count += 1
        if count >= max_files:
            break

    print(f"Downloaded {count} TED-LIUM talks to {out_dir}")


def download_mls_spanish(max_files: int = 5):
    """Download MLS Spanish test chapters (concatenated segments)."""
    from datasets import load_dataset

    out_dir = DATASETS_DIR / "mls-spanish"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MLS Spanish test split (streaming)...")
    ds = load_dataset("facebook/multilingual_librispeech", "spanish", split="test", streaming=True)

    # Group segments by chapter
    chapters = {}
    print("  Scanning segments by chapter...")
    for i, sample in enumerate(ds):
        cid = sample.get("chapter_id", 0)
        if cid not in chapters:
            chapters[cid] = {"arrays": [], "texts": [], "sr": sample["audio"]["sampling_rate"]}
        chapters[cid]["arrays"].append(np.array(sample["audio"]["array"], dtype=np.float32))
        chapters[cid]["texts"].append(sample.get("transcript", sample.get("text", "")))

        # Status update every 500 segments
        if (i + 1) % 500 == 0:
            print(f"    Scanned {i+1} segments across {len(chapters)} chapters...")

    print(f"  Found {len(chapters)} chapters from {i+1} segments")

    # Sort chapters by duration (longest first)
    chapter_durations = []
    for cid, data in chapters.items():
        total_samples = sum(len(a) for a in data["arrays"])
        dur = total_samples / data["sr"]
        chapter_durations.append((dur, cid))
    chapter_durations.sort(reverse=True)

    # Save the longest chapters
    count = 0
    for dur, cid in chapter_durations:
        if dur < 60:
            continue  # Skip short chapters

        fid = f"chapter_{cid}"
        wav_path = out_dir / f"{fid}.wav"
        ref_path = out_dir / f"{fid}.ref.txt"
        meta_path = out_dir / f"{fid}.meta.txt"

        if wav_path.exists():
            print(f"  {fid} already exists, skipping")
            count += 1
            if count >= max_files:
                break
            continue

        data = chapters[cid]
        print(f"  Saving {fid}: {dur:.0f}s ({dur/60:.1f}m), {len(data['arrays'])} segments")

        # Concatenate audio segments
        combined = np.concatenate(data["arrays"])
        sf.write(str(wav_path), combined, data["sr"])

        # Concatenate texts
        ref_path.write_text(" ".join(data["texts"]))

        # Save metadata
        meta_path.write_text(
            f"duration_secs: {dur:.1f}\nsource: MLS Spanish\nchapter_id: {cid}\n"
            f"segments: {len(data['arrays'])}\nlanguage: es\n"
        )

        count += 1
        if count >= max_files:
            break

    print(f"Downloaded {count} MLS Spanish chapters to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    parser.add_argument("--dataset", choices=["tedlium", "mls-spanish", "all"], default="all")
    parser.add_argument("--max-files", type=int, default=5)
    args = parser.parse_args()

    if args.dataset in ("tedlium", "all"):
        download_tedlium(args.max_files)

    if args.dataset in ("mls-spanish", "all"):
        download_mls_spanish(args.max_files)


if __name__ == "__main__":
    main()
