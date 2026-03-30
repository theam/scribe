import ArgumentParser
import AVFoundation
import FluidAudio
import Foundation

struct Stream: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Stream live audio transcription from microphone."
    )

    @Option(name: .long, help: "Output format: text or jsonl.")
    var format: StreamOutputFormat = .text

    @Option(name: .shortAndLong, help: "Also save output to file.")
    var output: String?

    @Flag(name: .long, help: "Show status information.")
    var verbose: Bool = false

    func run() async throws {
        if verbose { log("Initializing streaming ASR (Parakeet)...") }

        // Use the library's streaming config (11s chunks + 1s hypothesis updates)
        let streamManager = SlidingWindowAsrManager(config: .streaming)
        try await streamManager.start(source: .microphone)

        // Set up microphone capture
        let audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        if verbose {
            log("Microphone: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount) ch")
        }

        let startTime = Date()
        var outputFile: FileHandle? = nil
        var lastConfirmedText = ""
        var lastVolatileText = ""

        if let outputPath = output {
            FileManager.default.createFile(atPath: outputPath, contents: nil)
            outputFile = FileHandle(forWritingAtPath: outputPath)
        }

        signal(SIGINT) { _ in
            FileHandle.standardError.write(Data("\n".utf8))
            Darwin.exit(0)
        }

        // Install tap on microphone
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

            let elapsed = Date().timeIntervalSince(startTime)

            if update.isConfirmed {
                // Confirmed: stable, final text — print as a permanent line
                guard text != lastConfirmedText else { continue }
                lastConfirmedText = text
                lastVolatileText = "" // reset volatile tracking

                let line = formatLine(text: text, elapsed: elapsed, confirmed: true)
                print(line)
                fflush(stdout)

                if let file = outputFile {
                    file.write(Data((line + "\n").utf8))
                }
            } else {
                // Volatile: hypothesis, may change — show as ephemeral line
                guard text != lastVolatileText else { continue }
                lastVolatileText = text

                switch format {
                case .text:
                    // Overwrite current line with \r for live feel
                    let ts = formatTimestamp(elapsed)
                    let preview = String(text.suffix(80))
                    let line = "\r[\(ts)] \(preview)"
                    FileHandle.standardError.write(Data(line.utf8))
                case .jsonl:
                    // In JSONL mode, emit volatile updates too (marked as unconfirmed)
                    let line = formatLine(text: text, elapsed: elapsed, confirmed: false)
                    print(line)
                    fflush(stdout)
                }
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

        if let outputPath = output {
            FileHandle.standardError.write(Data("Saved to: \(outputPath)\n".utf8))
        }

        outputFile?.closeFile()
    }

    private func formatLine(text: String, elapsed: Double, confirmed: Bool) -> String {
        switch format {
        case .text:
            let ts = formatTimestamp(elapsed)
            return "[\(ts)] \(text)"
        case .jsonl:
            let jsonObj: [String: Any] = [
                "time": round(elapsed * 10) / 10,
                "text": text,
                "confirmed": confirmed,
            ]
            if let data = try? JSONSerialization.data(withJSONObject: jsonObj, options: [.sortedKeys]),
               let jsonStr = String(data: data, encoding: .utf8) {
                return jsonStr
            }
            return "{\"text\":\"\(text)\"}"
        }
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
