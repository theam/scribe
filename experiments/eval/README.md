# scribe Accuracy Evaluation

Deterministic evaluation of scribe transcription quality against standard ASR benchmarks.

## Datasets

| Tier | Dataset | Description | Source |
|------|---------|-------------|--------|
| Clean EN | [TED-LIUM 3](https://huggingface.co/datasets/distil-whisper/tedlium-long-form) | Single-speaker TED talks, professional audio | distil-whisper/tedlium-long-form |
| Clean ES | [MLS Spanish](https://huggingface.co/datasets/facebook/multilingual_librispeech) | Single-speaker audiobooks (LibriVox) | facebook/multilingual_librispeech |
| Hard EN | [Earnings-21](https://huggingface.co/datasets/Revai/earnings21) | Multi-speaker earnings calls, phone audio | Revai/earnings21 |

### Selection criteria

- **Transcription style**: Cleaned/professional — not verbatim with filler words (verbatim datasets inflate WER by ~80pp due to style mismatch, not quality)
- **Duration**: Files >5 minutes (representative of real usage)
- **License**: Free, no registration required
- **Reproducibility**: All datasets downloadable from HuggingFace

## Results (scribe v0.2.1)

| Dataset | Files | Audio | WER | Speed |
|---------|-------|-------|-----|-------|
| TED-LIUM 3 | 5 | 86m | **6.0%** | 138x real-time |
| MLS Spanish | 3 | 249m | **8.5%** | 80x real-time |
| Earnings-21 | 5 | 117m | **12.9%** | 135x real-time |

Raw per-file results in `results/scribe-0.2.1-*.csv`.

## Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets (~500MB)
python3 download_datasets.py

# Run evaluation (Earnings-21 auto-downloads)
python3 eval.py

# Single dataset
python3 eval.py --dataset tedlium
python3 eval.py --dataset mls-spanish
python3 eval.py --dataset earnings21

# Generate summary from CSV results
python3 summarize.py
```

## Methodology

- **WER computation**: [jiwer](https://github.com/jitsi/jiwer) library (industry standard)
- **Text normalization**: lowercase, remove punctuation, collapse whitespace (applied to both reference and hypothesis)
- **Determinism**: same audio = identical WER across runs (only wall-clock time varies)
- **No cherry-picking**: all files in each dataset tier are evaluated; aggregate WER is weighted by word count
