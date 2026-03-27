import Foundation
import SpeakerKit
import WhisperKit

/// Formats transcription results into various output formats.
enum OutputFormatter {

    static func format(
        results: [TranscriptionResult],
        diarization: DiarizationResult?,
        format: OutputFormat
    ) -> String {
        // If we have diarization, merge speaker info into segments
        let speakerSegments: [[SpeakerSegment]]?
        if let diarization = diarization {
            speakerSegments = diarization.addSpeakerInfo(to: results)
        } else {
            speakerSegments = nil
        }

        switch format {
        case .txt:
            return formatText(results: results, speakerSegments: speakerSegments)
        case .json:
            return formatJSON(results: results, speakerSegments: speakerSegments, hasDiarization: diarization != nil)
        case .srt:
            return formatSRT(results: results, speakerSegments: speakerSegments)
        case .vtt:
            return formatVTT(results: results, speakerSegments: speakerSegments)
        }
    }

    // MARK: - Plain Text

    private static func formatText(results: [TranscriptionResult], speakerSegments: [[SpeakerSegment]]?) -> String {
        if let speakerSegments = speakerSegments {
            return formatTextWithSpeakers(speakerSegments: speakerSegments)
        }
        return formatTextPlain(results: results)
    }

    private static func formatTextPlain(results: [TranscriptionResult]) -> String {
        var lines: [String] = []
        for result in results {
            for segment in result.segments {
                let timestamp = formatTimestamp(segment.start)
                lines.append("[\(timestamp)] \(cleanText(segment.text))")
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func formatTextWithSpeakers(speakerSegments: [[SpeakerSegment]]) -> String {
        var lines: [String] = []
        for group in speakerSegments {
            for segment in group {
                let timestamp = formatTimestamp(segment.startTime)
                let speaker = speakerLabel(segment.speaker)
                let text = cleanText(segment.transcription?.text ?? "") ?? ""
                if !text.isEmpty {
                    lines.append("[\(timestamp)] \(speaker): \(text)")
                }
            }
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - JSON

    private static func formatJSON(results: [TranscriptionResult], speakerSegments: [[SpeakerSegment]]?, hasDiarization: Bool) -> String {
        var segments: [[String: Any]] = []

        if let speakerSegments = speakerSegments {
            for group in speakerSegments {
                for segment in group {
                    var dict: [String: Any] = [
                        "start": segment.startTime,
                        "end": segment.endTime,
                        "speaker": speakerLabel(segment.speaker),
                    ]
                    if let transcription = segment.transcription {
                        dict["text"] = transcription.text.trimmingCharacters(in: .whitespaces)
                    }
                    if !segment.speakerWords.isEmpty {
                        dict["words"] = segment.speakerWords.map { word in
                            var w: [String: Any] = [
                                "start": word.wordTiming.start,
                                "end": word.wordTiming.end,
                                "text": word.wordTiming.word,
                            ]
                            w["speaker"] = speakerLabel(word.speaker)
                            return w
                        }
                    }
                    segments.append(dict)
                }
            }
        } else {
            for result in results {
                for segment in result.segments {
                    var dict: [String: Any] = [
                        "start": segment.start,
                        "end": segment.end,
                        "text": cleanText(segment.text),
                    ]
                    if let words = segment.words, !words.isEmpty {
                        dict["words"] = words.map { word in
                            [
                                "start": word.start,
                                "end": word.end,
                                "text": word.word,
                            ] as [String: Any]
                        }
                    }
                    segments.append(dict)
                }
            }
        }

        let totalDuration = results.last?.segments.last?.end ?? 0

        let output: [String: Any] = [
            "metadata": [
                "duration": totalDuration,
                "diarization": hasDiarization,
            ],
            "segments": segments,
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys]),
              let string = String(data: data, encoding: .utf8) else {
            return "{\"error\": \"Failed to serialize JSON\"}"
        }

        return string
    }

    // MARK: - SRT

    private static func formatSRT(results: [TranscriptionResult], speakerSegments: [[SpeakerSegment]]?) -> String {
        var lines: [String] = []
        var index = 1

        if let speakerSegments = speakerSegments {
            for group in speakerSegments {
                for segment in group {
                    let text = cleanText(segment.transcription?.text ?? "") ?? ""
                    guard !text.isEmpty else { continue }
                    let start = formatSRTTimestamp(segment.startTime)
                    let end = formatSRTTimestamp(segment.endTime)
                    let speaker = speakerLabel(segment.speaker)
                    lines.append("\(index)")
                    lines.append("\(start) --> \(end)")
                    lines.append("[\(speaker)] \(text)")
                    lines.append("")
                    index += 1
                }
            }
        } else {
            for result in results {
                for segment in result.segments {
                    let start = formatSRTTimestamp(segment.start)
                    let end = formatSRTTimestamp(segment.end)
                    let text = cleanText(segment.text)
                    lines.append("\(index)")
                    lines.append("\(start) --> \(end)")
                    lines.append(text)
                    lines.append("")
                    index += 1
                }
            }
        }

        return lines.joined(separator: "\n")
    }

    // MARK: - VTT

    private static func formatVTT(results: [TranscriptionResult], speakerSegments: [[SpeakerSegment]]?) -> String {
        var lines: [String] = ["WEBVTT", ""]

        if let speakerSegments = speakerSegments {
            for group in speakerSegments {
                for segment in group {
                    let text = cleanText(segment.transcription?.text ?? "") ?? ""
                    guard !text.isEmpty else { continue }
                    let start = formatVTTTimestamp(segment.startTime)
                    let end = formatVTTTimestamp(segment.endTime)
                    let speaker = speakerLabel(segment.speaker)
                    lines.append("\(start) --> \(end)")
                    lines.append("<v \(speaker)>\(text)")
                    lines.append("")
                }
            }
        } else {
            for result in results {
                for segment in result.segments {
                    let start = formatVTTTimestamp(segment.start)
                    let end = formatVTTTimestamp(segment.end)
                    let text = cleanText(segment.text)
                    lines.append("\(start) --> \(end)")
                    lines.append(text)
                    lines.append("")
                }
            }
        }

        return lines.joined(separator: "\n")
    }

    // MARK: - Helpers

    /// Strip Whisper special tokens from text (e.g., <|startoftranscript|>, <|en|>, <|0.00|>)
    private static func cleanText(_ text: String) -> String {
        // Remove all <|...|> tokens
        let pattern = "<\\|[^|]*\\|>"
        let cleaned = text.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        return cleaned.trimmingCharacters(in: .whitespaces)
    }

    private static func speakerLabel(_ info: SpeakerInfo) -> String {
        switch info {
        case .speakerId(let id):
            return "Speaker \(id + 1)"
        case .multiple(let ids):
            return ids.map { "Speaker \($0 + 1)" }.joined(separator: "/")
        case .noMatch:
            return "Unknown"
        }
    }

    private static func formatTimestamp(_ seconds: Float) -> String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        if hours > 0 {
            return String(format: "%02d:%02d:%02d", hours, minutes, secs)
        }
        return String(format: "%02d:%02d", minutes, secs)
    }

    private static func formatSRTTimestamp(_ seconds: Float) -> String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        let millis = Int((seconds - Float(totalSeconds)) * 1000)
        return String(format: "%02d:%02d:%02d,%03d", hours, minutes, secs, millis)
    }

    private static func formatVTTTimestamp(_ seconds: Float) -> String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        let millis = Int((seconds - Float(totalSeconds)) * 1000)
        return String(format: "%02d:%02d:%02d.%03d", hours, minutes, secs, millis)
    }
}
