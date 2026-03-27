import ArgumentParser
import Foundation
import SpeakerKit
import WhisperKit

struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Transcribe an audio file."
    )

    @Argument(help: "Path to the audio file to transcribe.")
    var audioFile: String

    @Flag(name: .long, help: "Enable speaker diarization.")
    var diarize: Bool = false

    @Option(name: .long, help: "Number of speakers (helps diarization accuracy).")
    var speakers: Int?

    @Option(name: .long, help: "Output format: txt, json, srt, vtt.")
    var format: OutputFormat = .txt

    @Option(name: .long, help: "Whisper model to use (e.g., large-v3, large-v3-turbo, small, tiny).")
    var model: String = "large-v3-v20240930_turbo_632MB"

    @Option(name: .long, help: "Language code (auto-detect if not specified).")
    var language: String?

    @Option(name: .shortAndLong, help: "Output file path (default: stdout).")
    var output: String?

    @Flag(name: .long, help: "Show progress and timing information.")
    var verbose: Bool = false

    func run() async throws {
        let fileURL = URL(fileURLWithPath: audioFile)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw ValidationError("File not found: \(audioFile)")
        }

        if verbose {
            log("Loading model '\(model)'...")
        }

        // Initialize WhisperKit
        let config = WhisperKitConfig(
            model: model,
            verbose: verbose,
            logLevel: verbose ? .debug : .error
        )
        let whisperKit = try await WhisperKit(config)

        if verbose {
            log("Transcribing '\(audioFile)'...")
        }

        // Transcribe
        let startTime = Date()
        var options = DecodingOptions()
        if let lang = language {
            options.language = lang
        }
        options.wordTimestamps = true

        let results = try await whisperKit.transcribe(audioPath: fileURL.path, decodeOptions: options)
        let transcribeElapsed = Date().timeIntervalSince(startTime)

        if verbose {
            log(String(format: "Transcription completed in %.1fs", transcribeElapsed))
        }

        // Diarize if requested
        var diarizationResult: DiarizationResult? = nil
        if diarize {
            if verbose {
                log("Diarizing speakers...")
            }
            let diarizeStart = Date()
            diarizationResult = try await Diarizer.diarize(
                audioPath: fileURL.path,
                numSpeakers: speakers,
                verbose: verbose
            )
            let diarizeElapsed = Date().timeIntervalSince(diarizeStart)
            if verbose {
                log(String(format: "Diarization completed in %.1fs (%d speakers detected)",
                           diarizeElapsed, diarizationResult?.speakerCount ?? 0))
            }
        }

        // Format output
        let formatted = OutputFormatter.format(
            results: results,
            diarization: diarizationResult,
            format: format
        )

        // Write output
        if let outputPath = output {
            try formatted.write(toFile: outputPath, atomically: true, encoding: String.Encoding.utf8)
            if verbose {
                log("Written to \(outputPath)")
            }
        } else {
            print(formatted)
        }

        if verbose {
            let totalElapsed = Date().timeIntervalSince(startTime)
            log(String(format: "Total time: %.1fs", totalElapsed))
        }
    }

    /// Write to stderr so it doesn't mix with stdout output.
    private func log(_ message: String) {
        FileHandle.standardError.write(Data("[scribe] \(message)\n".utf8))
    }
}

// MARK: - Output Format

enum OutputFormat: String, ExpressibleByArgument, CaseIterable {
    case txt, json, srt, vtt
}
