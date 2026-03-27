# scribe

State-of-the-art local audio transcription with speaker diarization for macOS.

100% local. No cloud. No API keys. No data leaves your machine.

## Features

- **Transcription** — Accurate speech-to-text powered by OpenAI Whisper (via WhisperKit)
- **Speaker diarization** — Identify who said what, powered by pyannote (via SpeakerKit)
- **Apple Silicon optimized** — Runs on CoreML and the Apple Neural Engine
- **Multiple output formats** — Plain text, JSON (with word timestamps), SRT, VTT
- **99 languages** — Supports all languages that Whisper supports
- **Fast** — Processes audio faster than real-time on Apple Silicon

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

### Specify number of speakers (improves accuracy)

```bash
scribe transcribe meeting.wav --diarize --speakers 4
```

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

### Choose a model

```bash
scribe transcribe meeting.wav --model large-v3         # best accuracy
scribe transcribe meeting.wav --model large-v3-turbo   # best speed/accuracy (default)
scribe transcribe meeting.wav --model small             # fastest, lower accuracy
```

### Force a language

```bash
scribe transcribe meeting.wav --language es    # Spanish
scribe transcribe meeting.wav --language fr    # French
```

### Manage models

```bash
scribe models list                       # see available/downloaded models
scribe models download large-v3-turbo    # download a model
scribe models remove large-v3            # remove a downloaded model
```

## Output Examples

### Plain text with diarization

```
[00:00:12] Speaker 1: Hello everyone, welcome to the meeting.
[00:00:18] Speaker 2: Thanks for joining. Let's start with the agenda.
```

### JSON

```json
{
  "metadata": {
    "duration": 3612.5,
    "diarization": true
  },
  "segments": [
    {
      "start": 12.0,
      "end": 16.5,
      "text": "Hello everyone, welcome to the meeting.",
      "speaker": "Speaker 1",
      "words": [
        { "start": 12.0, "end": 12.4, "text": "Hello" }
      ]
    }
  ]
}
```

## Requirements

- macOS 14 (Sonoma) or later
- Apple Silicon (M1 or later) recommended — Intel Macs fall back to CPU inference

## Models

The default model (`large-v3-turbo`, ~632MB) is downloaded automatically on first use. Other models can be downloaded with `scribe models download`.

| Model | Size | Speed | Accuracy | Languages |
|-------|------|-------|----------|-----------|
| tiny | ~75MB | Fastest | Lower | 99 |
| base | ~142MB | Very fast | Fair | 99 |
| small | ~466MB | Fast | Good | 99 |
| medium | ~1.5GB | Moderate | Very good | 99 |
| large-v3-turbo | ~632MB | **Fast** | **Very good** | 99 |
| large-v3 | ~3.1GB | Slower | Best | 99 |

## Acknowledgments

scribe is built on the shoulders of excellent open-source projects:

- **[OpenAI Whisper](https://github.com/openai/whisper)** (Apache 2.0) — The speech recognition model that powers transcription
- **[WhisperKit](https://github.com/argmaxinc/WhisperKit)** (MIT) by Argmax — CoreML implementation of Whisper for Apple Silicon
- **[SpeakerKit](https://github.com/argmaxinc/WhisperKit)** (MIT) by Argmax — CoreML speaker diarization
- **[pyannote.audio](https://github.com/pyannote/pyannote-audio)** (MIT) by Herve Bredin — The diarization model architecture that SpeakerKit builds on
- **[swift-argument-parser](https://github.com/apple/swift-argument-parser)** (Apache 2.0) by Apple — CLI argument parsing

## License

Apache 2.0 — Copyright 2026 The Agile Monkeys Inc. See [LICENSE](LICENSE).
