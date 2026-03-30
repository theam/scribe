import ArgumentParser
import FluidAudio
import Foundation

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

        // Load audio
        if verbose { log("Loading audio...") }
        let converter = AudioConverter()
        let samples = try converter.resampleAudioFile(fileURL)
        let duration = Double(samples.count) / 16000.0

        if verbose { log(String(format: "Audio loaded: %.0fs (%.1f min)", duration, duration / 60)) }

        // Transcribe with Parakeet TDT v3
        if verbose { log("Loading ASR model (Parakeet TDT v3)...") }
        let asrModels = try await AsrModels.downloadAndLoad(version: .v3)
        let asrManager = AsrManager()
        try await asrManager.initialize(models: asrModels)

        if verbose { log("Transcribing '\(audioFile)'...") }
        let startTime = Date()
        let asrResult = try await asrManager.transcribe(samples, source: .system)
        let transcribeElapsed = Date().timeIntervalSince(startTime)

        if verbose {
            log(String(format: "Transcription completed in %.1fs (%.1fx real-time)",
                       transcribeElapsed, duration / transcribeElapsed))
        }

        // Diarize if requested
        var diarizationSegments: [DiarizationSegment]? = nil
        if diarize {
            if verbose { log("Loading diarization model...") }
            let diarizeStart = Date()

            // High-quality config: stepRatio=0.1, minSegmentDuration=0, lower clustering threshold
            var offlineConfig = OfflineDiarizerConfig(
                clusteringThreshold: 0.7,
                segmentationStepRatio: 0.1,
                minSegmentDuration: 0.0
            )
            if let n = speakers {
                offlineConfig.clustering.numSpeakers = n
            }
            let offlineManager = OfflineDiarizerManager(config: offlineConfig)
            try await offlineManager.prepareModels()

            if verbose { log("Diarizing speakers...") }
            let diarResult = try await offlineManager.process(audio: samples)

            let speakerCount = Set(diarResult.segments.map { $0.speakerId }).count
            let diarizeElapsed = Date().timeIntervalSince(diarizeStart)

            diarizationSegments = diarResult.segments.map { seg in
                DiarizationSegment(
                    speaker: seg.speakerId,
                    start: Double(seg.startTimeSeconds),
                    end: Double(seg.endTimeSeconds)
                )
            }

            if verbose {
                log(String(format: "Diarization completed in %.1fs (%d speakers detected)",
                           diarizeElapsed, speakerCount))
            }
        }

        // Format output
        let formatted = OutputFormatter.format(
            asrResult: asrResult,
            diarization: diarizationSegments,
            duration: duration,
            format: format
        )

        // Write output
        if let outputPath = output {
            try formatted.write(toFile: outputPath, atomically: true, encoding: String.Encoding.utf8)
            if verbose { log("Written to \(outputPath)") }
        } else {
            print(formatted, terminator: "")
            fflush(stdout)
        }

        if verbose {
            let totalElapsed = Date().timeIntervalSince(startTime)
            log(String(format: "Total time: %.1fs", totalElapsed))
        }

        await asrManager.cleanup()
    }

    private func log(_ message: String) {
        FileHandle.standardError.write(Data("[scribe] \(message)\n".utf8))
    }
}

// MARK: - Output Format

enum OutputFormat: String, ExpressibleByArgument, CaseIterable, Sendable {
    case txt, json, srt, vtt
}

// MARK: - Diarization Segment (internal)

struct DiarizationSegment: Sendable {
    let speaker: String
    let start: Double
    let end: Double
}
