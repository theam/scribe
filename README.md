# scribe

State-of-the-art local audio transcription with speaker diarization for macOS.

100% local. No cloud. No API keys. No data leaves your machine.

## Features

- **Transcription** — Accurate speech-to-text powered by NVIDIA Parakeet TDT v3 (via FluidAudio CoreML)
- **Speaker diarization** — Identify who said what, powered by pyannote (via FluidAudio CoreML)
- **Apple Silicon optimized** — Runs on CoreML and the Apple Neural Engine at 130x real-time
- **Multiple output formats** — Plain text, JSON (with word timestamps), SRT, VTT
- **25 European languages** — English, Spanish, French, German, Italian, Portuguese, Russian, and more
- **Fast** — Transcribes a 4-minute recording in under 2 seconds

## Install

```bash
brew install theam/tap/scribe
```

Or build from source:

```bash
git clone https://github.com/theam/scribe.git
cd scribe
swift build -c release
cp .build/release/scribe /usr/local/bin/
```

## Usage

### Basic transcription

```bash
scribe transcribe meeting.wav
```

### With speaker diarization

```bash
scribe transcribe meeting.wav --diarize
```

### Specify number of speakers (recommended for best diarization)

```bash
scribe transcribe meeting.wav --diarize --speakers 4
```

> **Tip:** Providing the expected number of speakers with `--speakers` significantly improves diarization accuracy. Without it, the automatic speaker count detection works well for most recordings but may slightly over- or under-segment when voices are similar. If you know how many people were in the meeting, always pass `--speakers`.

### Output formats

```bash
scribe transcribe meeting.wav --format txt    # plain text (default)
scribe transcribe meeting.wav --format json   # structured JSON with word timestamps
scribe transcribe meeting.wav --format srt    # SRT subtitles
scribe transcribe meeting.wav --format vtt    # WebVTT subtitles
```

### Save to file

```bash
scribe transcribe meeting.wav --format json --output transcript.json
```

### Force a language

```bash
scribe transcribe meeting.wav --language es    # Spanish
scribe transcribe meeting.wav --language fr    # French
```

### Live streaming from microphone

```bash
scribe stream                      # multilingual (25 languages, ~11s latency)
scribe stream --engine nemotron    # English-only, low latency (~560ms), with punctuation
scribe stream --format jsonl       # JSONL output (one JSON object per line)
scribe stream --output meeting.txt # save to file while streaming
```

> **Note on streaming engines:**
> - **Default** (Parakeet TDT v3): Supports all 25 languages including Spanish. Higher latency (~11s for confirmed text) because it uses a batch model in sliding windows. Live preview appears on screen while speaking.
> - **Nemotron** (`--engine nemotron`): English-only but much faster (~560ms latency). Includes punctuation and capitalization. Recommended for English-only meetings.

### Pre-download models

```bash
scribe models download all    # download ASR + diarization models for offline use
```

## Output Examples

### Plain text with diarization

```
[00:03] Speaker 1: Hello, how are you?
[00:06] Speaker 1: I forgot a few points.
[00:27] Speaker 2: Let's see if Claude is right about you.
[00:32] Speaker 3: Oh my gosh, here comes the song. My favorite.
```

### JSON

```json
{
  "metadata": {
    "duration": 226.1,
    "diarization": true
  },
  "segments": [
    {
      "start": 3.2,
      "end": 4.7,
      "text": "Hello, how are you?",
      "speaker": "Speaker 1",
      "words": [
        { "start": 3.2, "end": 3.6, "text": "Hello," },
        { "start": 3.6, "end": 3.9, "text": "how" },
        { "start": 3.9, "end": 4.2, "text": "are" },
        { "start": 4.2, "end": 4.7, "text": "you?" }
      ]
    }
  ]
}
```

## Accuracy

Measured on standard evaluation datasets using [jiwer](https://github.com/jitsi/jiwer) (Word Error Rate — lower is better). Results are deterministic and reproducible.

| Dataset | Description | WER | Speed |
|---------|-------------|-----|-------|
| [TED-LIUM 3](https://huggingface.co/datasets/distil-whisper/tedlium-long-form) | TED talks — clean English, single speaker | **6.0%** | 138x real-time |
| [MLS Spanish](https://huggingface.co/datasets/facebook/multilingual_librispeech) | Audiobooks — clean Spanish, single speaker | **8.5%** | 80x real-time |
| [Earnings-21](https://huggingface.co/datasets/Revai/earnings21) | Earnings calls — noisy English, 5-10 speakers | **12.9%** | 135x real-time |

These numbers are competitive with commercial cloud transcription services:

| Service | Typical WER | Price | Local? |
|---------|------------|-------|--------|
| **scribe** | **6–13%** | **Free** | **Yes** |
| AssemblyAI | 5.6% (avg) | $0.15–0.45/hr | No |
| Deepgram Nova-3 | 5.3% (batch) | $0.39–0.46/hr | No |
| Otter.ai | 12–19% | ~$1/hr | No |

> Eval methodology, datasets, and raw CSV results: [`experiments/eval/`](https://github.com/theam/scribe/tree/main/experiments/eval)

## Performance

Tested on Apple Silicon (M-series):

| Task | Speed | Example |
|------|-------|---------|
| Transcription only | ~130x real-time | 4-min file in 1.7s |
| Transcription + diarization | ~30x real-time | 4-min file in 7.5s |
| Live streaming (Nemotron) | ~560ms latency | English, with punctuation |
| Live streaming (default) | ~11s latency | 25 languages |

Models are downloaded automatically on first use (~600MB per model). First run may take a minute.

## Requirements

- macOS 14 (Sonoma) or later
- Apple Silicon (M1 or later)

## Supported Languages

Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish, Ukrainian.

## Acknowledgments

scribe is built on the shoulders of excellent open-source projects:

- **[NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** (CC-BY-4.0) — Speech recognition models (TDT v3 for batch, Nemotron for streaming)
- **[FluidAudio](https://github.com/FluidInference/FluidAudio)** (Apache 2.0) by FluidInference — CoreML speech processing SDK for Apple Silicon
- **[pyannote.audio](https://github.com/pyannote/pyannote-audio)** (MIT) by Herve Bredin — The diarization model architecture
- **[swift-argument-parser](https://github.com/apple/swift-argument-parser)** (Apache 2.0) by Apple — CLI argument parsing

## License

Apache 2.0 — Copyright 2026 The Agile Monkeys Inc. See [LICENSE](LICENSE).
