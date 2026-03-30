import FluidAudio
import Foundation

/// Formats transcription results into various output formats.
enum OutputFormatter {

    static func format(
        asrResult: ASRResult,
        diarization: [DiarizationSegment]?,
        duration: Double,
        format: OutputFormat
    ) -> String {
        // Build segments from token timings or fall back to full text
        let segments = buildSegments(asrResult: asrResult, diarization: diarization)

        switch format {
        case .txt:
            return formatText(segments: segments)
        case .json:
            return formatJSON(segments: segments, duration: duration, hasDiarization: diarization != nil)
        case .srt:
            return formatSRT(segments: segments)
        case .vtt:
            return formatVTT(segments: segments)
        }
    }

    // MARK: - Segment Building

    private struct Word {
        let start: Double
        let end: Double
        let text: String
    }

    private struct Segment {
        let start: Double
        let end: Double
        let text: String
        let speaker: String?
        let words: [Word]
    }

    /// Merge BPE/SentencePiece tokens into whole words.
    /// Convention: a token starting with a space begins a new word; others continue the previous.
    private static func mergeTokensIntoWords(timings: [TokenTiming]) -> [Word] {
        var words: [Word] = []
        var currentText = ""
        var wordStart: Double = 0
        var wordEnd: Double = 0

        for timing in timings {
            let token = timing.token
            if token.trimmingCharacters(in: .whitespaces).isEmpty { continue }

            let startsNewWord = token.hasPrefix(" ") || words.isEmpty && currentText.isEmpty

            if startsNewWord && !currentText.isEmpty {
                // Flush previous word
                let trimmed = currentText.trimmingCharacters(in: .whitespaces)
                if !trimmed.isEmpty {
                    words.append(Word(start: wordStart, end: wordEnd, text: trimmed))
                }
                currentText = ""
            }

            if currentText.isEmpty {
                wordStart = timing.startTime
            }
            currentText += token
            wordEnd = timing.endTime
        }

        // Flush last word
        let trimmed = currentText.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty {
            words.append(Word(start: wordStart, end: wordEnd, text: trimmed))
        }

        return words
    }

    private static func buildSegments(asrResult: ASRResult, diarization: [DiarizationSegment]?) -> [Segment] {
        guard let timings = asrResult.tokenTimings, !timings.isEmpty else {
            return [Segment(
                start: 0,
                end: asrResult.duration,
                text: cleanText(asrResult.text),
                speaker: nil,
                words: []
            )]
        }

        let allWords = mergeTokensIntoWords(timings: timings)

        // Group words into sentence-like segments (split on sentence-ending punctuation)
        var segments: [Segment] = []
        var currentWords: [Word] = []
        var segStart: Double = 0

        for word in allWords {
            if currentWords.isEmpty {
                segStart = word.start
            }
            currentWords.append(word)

            let endsWithPunctuation = word.text.hasSuffix(".") || word.text.hasSuffix("?") || word.text.hasSuffix("!")
            if endsWithPunctuation || currentWords.count >= 30 {
                let text = currentWords.map { $0.text }.joined(separator: " ")
                let segEnd = currentWords.last!.end

                let speaker: String? = diarization.flatMap { diar in
                    findSpeaker(at: segStart, in: diar)
                }

                segments.append(Segment(
                    start: segStart,
                    end: segEnd,
                    text: cleanText(text),
                    speaker: speaker,
                    words: currentWords
                ))
                currentWords = []
            }
        }

        // Flush remaining
        if !currentWords.isEmpty {
            let text = currentWords.map { $0.text }.joined(separator: " ")
            let segEnd = currentWords.last!.end
            let speaker: String? = diarization.flatMap { diar in
                findSpeaker(at: segStart, in: diar)
            }
            segments.append(Segment(
                start: segStart,
                end: segEnd,
                text: cleanText(text),
                speaker: speaker,
                words: currentWords
            ))
        }

        return segments
    }

    // MARK: - Plain Text

    private static func formatText(segments: [Segment]) -> String {
        var lines: [String] = []
        for seg in segments {
            let ts = formatTimestamp(seg.start)
            if let speaker = seg.speaker {
                lines.append("[\(ts)] \(speakerLabel(speaker)): \(seg.text)")
            } else {
                lines.append("[\(ts)] \(seg.text)")
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    // MARK: - JSON

    private static func formatJSON(segments: [Segment], duration: Double, hasDiarization: Bool) -> String {
        var jsonSegments: [[String: Any]] = []

        for seg in segments {
            var dict: [String: Any] = [
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            ]
            if let speaker = seg.speaker {
                dict["speaker"] = speakerLabel(speaker)
            }
            if !seg.words.isEmpty {
                dict["words"] = seg.words.map { w in
                    [
                        "start": w.start,
                        "end": w.end,
                        "text": w.text,
                    ] as [String: Any]
                }
            }
            jsonSegments.append(dict)
        }

        let output: [String: Any] = [
            "metadata": [
                "duration": duration,
                "diarization": hasDiarization,
            ],
            "segments": jsonSegments,
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys]),
              let string = String(data: data, encoding: .utf8) else {
            return "{\"error\": \"Failed to serialize JSON\"}"
        }

        return string + "\n"
    }

    // MARK: - SRT

    private static func formatSRT(segments: [Segment]) -> String {
        var lines: [String] = []
        for (i, seg) in segments.enumerated() {
            let start = formatSRTTimestamp(seg.start)
            let end = formatSRTTimestamp(seg.end)
            let prefix = seg.speaker.map { "[\(speakerLabel($0))] " } ?? ""
            lines.append("\(i + 1)")
            lines.append("\(start) --> \(end)")
            lines.append("\(prefix)\(seg.text)")
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - VTT

    private static func formatVTT(segments: [Segment]) -> String {
        var lines: [String] = ["WEBVTT", ""]
        for seg in segments {
            let start = formatVTTTimestamp(seg.start)
            let end = formatVTTTimestamp(seg.end)
            lines.append("\(start) --> \(end)")
            if let speaker = seg.speaker {
                lines.append("<v \(speakerLabel(speaker))>\(seg.text)")
            } else {
                lines.append(seg.text)
            }
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Helpers

    private static func cleanText(_ text: String) -> String {
        let pattern = "<\\|[^|]*\\|>"
        let cleaned = text.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        return cleaned.trimmingCharacters(in: .whitespaces)
    }

    private static func speakerLabel(_ id: String) -> String {
        // FluidAudio returns IDs like "0", "1", "2" — map to "Speaker 1", etc.
        if let num = Int(id) {
            return "Speaker \(num + 1)"
        }
        return id
    }

    private static func findSpeaker(at time: Double, in diarization: [DiarizationSegment]) -> String? {
        return diarization.first(where: { time >= $0.start && time < $0.end })?.speaker
    }

    private static func formatTimestamp(_ seconds: Double) -> String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        if hours > 0 {
            return String(format: "%02d:%02d:%02d", hours, minutes, secs)
        }
        return String(format: "%02d:%02d", minutes, secs)
    }

    private static func formatSRTTimestamp(_ seconds: Double) -> String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        let millis = Int((seconds - Double(totalSeconds)) * 1000)
        return String(format: "%02d:%02d:%02d,%03d", hours, minutes, secs, millis)
    }

    private static func formatVTTTimestamp(_ seconds: Double) -> String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        let millis = Int((seconds - Double(totalSeconds)) * 1000)
        return String(format: "%02d:%02d:%02d.%03d", hours, minutes, secs, millis)
    }
}
