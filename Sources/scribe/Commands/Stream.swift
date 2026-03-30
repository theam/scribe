import ArgumentParser
import AVFoundation
import FluidAudio
import Foundation

struct Stream: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Stream live audio transcription from microphone or system audio."
    )

    @Option(name: .long, help: "Output format: text or jsonl.")
    var format: StreamOutputFormat = .text

    @Option(name: .shortAndLong, help: "Also save output to file.")
    var output: String?

    @Flag(name: .long, help: "Show status information.")
    var verbose: Bool = false

    func run() async throws {
        if verbose { log("Initializing streaming ASR (Parakeet)...") }

        // Low-latency config: 3s chunks for fast output, 1s hypothesis for immediate feedback
        let config = SlidingWindowAsrConfig(
            chunkSeconds: 3.0,
            hypothesisChunkSeconds: 1.0,
            leftContextSeconds: 3.0,
            rightContextSeconds: 0.5,
            minContextForConfirmation: 3.0,
            confirmationThreshold: 0.5
        )
        let streamManager = SlidingWindowAsrManager(config: config)
        try await streamManager.start(source: .microphone)

        // Set up microphone capture
        let audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        if verbose {
            log("Microphone: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount) ch")
        }

        // Track state for output
        let startTime = Date()
        var outputFile: FileHandle? = nil
        var utteranceCount = 0
        var lastEmittedText = ""

        if let outputPath = output {
            FileManager.default.createFile(atPath: outputPath, contents: nil)
            outputFile = FileHandle(forWritingAtPath: outputPath)
        }

        // Set up signal handler for clean Ctrl+C
        signal(SIGINT) { _ in
            FileHandle.standardError.write(Data("\n".utf8))
            Darwin.exit(0)
        }

        // Install tap on microphone — feed buffers to ASR
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            nonisolated(unsafe) let buf = buffer
            streamManager.streamAudio(buf)
        }

        try audioEngine.start()
        log("Listening... (press Ctrl+C to stop)")

        // Read transcription updates
        let updates = await streamManager.transcriptionUpdates
        for await update in updates {
            let text = update.text.trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { continue }

            // Skip if this is the same text we already emitted (dedup)
            guard text != lastEmittedText else { continue }
            lastEmittedText = text

            utteranceCount += 1
            let elapsed = Date().timeIntervalSince(startTime)

            let line: String
            switch format {
            case .text:
                let ts = formatTimestamp(elapsed)
                line = "[\(ts)] \(text)"
            case .jsonl:
                let jsonObj: [String: Any] = [
                    "time": round(elapsed * 10) / 10,
                    "text": text,
                    "confirmed": update.isConfirmed,
                ]
                if let data = try? JSONSerialization.data(withJSONObject: jsonObj, options: [.sortedKeys]),
                   let jsonStr = String(data: data, encoding: .utf8) {
                    line = jsonStr
                } else {
                    continue
                }
            }

            print(line)
            fflush(stdout)

            if let file = outputFile {
                file.write(Data((line + "\n").utf8))
            }
        }

        // Clean shutdown
        audioEngine.stop()
        inputNode.removeTap(onBus: 0)
        _ = try await streamManager.finish()

        let totalElapsed = Date().timeIntervalSince(startTime)

        FileHandle.standardError.write(Data("\n--- Stream ended ---\n".utf8))
        FileHandle.standardError.write(Data(String(format: "Duration: %dm %ds\n",
            Int(totalElapsed) / 60, Int(totalElapsed) % 60).utf8))
        FileHandle.standardError.write(Data("Utterances: \(utteranceCount)\n".utf8))

        if let outputPath = output {
            FileHandle.standardError.write(Data("Saved to: \(outputPath)\n".utf8))
        }

        outputFile?.closeFile()
    }

    private func log(_ message: String) {
        FileHandle.standardError.write(Data("[scribe] \(message)\n".utf8))
    }

    private func formatTimestamp(_ seconds: Double) -> String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        if hours > 0 {
            return String(format: "%02d:%02d:%02d", hours, minutes, secs)
        }
        return String(format: "%02d:%02d", minutes, secs)
    }
}

enum StreamOutputFormat: String, ExpressibleByArgument, CaseIterable, Sendable {
    case text, jsonl
}
