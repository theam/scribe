import ArgumentParser
import AVFoundation
import FluidAudio
import Foundation

/// Thread-safe state tracker for streaming output.
private actor StreamState {
    var lastPartialText = ""
    var lastOutputText = ""

    func shouldEmitPartial(_ text: String) -> Bool {
        guard text != lastPartialText else { return false }
        lastPartialText = text
        return true
    }

    func getNewText(_ fullTranscript: String) -> String? {
        let text = fullTranscript.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, text != lastOutputText else { return nil }

        let newText: String
        if text.hasPrefix(lastOutputText) {
            newText = String(text.dropFirst(lastOutputText.count)).trimmingCharacters(in: .whitespaces)
        } else {
            newText = text
        }

        guard !newText.isEmpty else { return nil }
        lastOutputText = text
        return newText
    }
}

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
        if verbose { log("Initializing streaming ASR (Nemotron 560ms)...") }

        let engine = StreamingAsrEngineFactory.create(.nemotron560ms)
        try await engine.loadModels()

        if verbose { log("Models loaded.") }

        let audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        if verbose {
            log("Microphone: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount) ch")
        }

        let startTime = Date()
        let state = StreamState()
        let fmt = format
        var outputFile: FileHandle? = nil

        if let outputPath = output {
            FileManager.default.createFile(atPath: outputPath, contents: nil)
            outputFile = FileHandle(forWritingAtPath: outputPath)
        }

        // Partial transcript callback — fires on every chunk (~560ms)
        await engine.setPartialTranscriptCallback { partial in
            let text = partial.trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { return }

            Task {
                guard await state.shouldEmitPartial(text) else { return }
                let elapsed = Date().timeIntervalSince(startTime)

                switch fmt {
                case .text:
                    let ts = formatStreamTimestamp(elapsed)
                    let preview = String(text.suffix(100))
                    FileHandle.standardError.write(Data("\r\u{1B}[K[\(ts)] \(preview)".utf8))
                case .jsonl:
                    let jsonObj: [String: Any] = [
                        "time": round(elapsed * 10) / 10,
                        "text": text,
                        "partial": true,
                    ]
                    if let data = try? JSONSerialization.data(withJSONObject: jsonObj, options: [.sortedKeys]),
                       let jsonStr = String(data: data, encoding: .utf8) {
                        print(jsonStr)
                        fflush(stdout)
                    }
                }
            }
        }

        signal(SIGINT) { _ in
            FileHandle.standardError.write(Data("\n".utf8))
            Darwin.exit(0)
        }

        // Install tap on microphone
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            nonisolated(unsafe) let buf = buffer
            do {
                try engine.appendAudio(buf)
            } catch {
                // Non-fatal
            }
        }

        try audioEngine.start()
        log("Listening... (press Ctrl+C to stop)")

        // Processing loop
        while true {
            try await engine.processBufferedAudio()

            let transcript = await engine.getPartialTranscript()
            if let newText = await state.getNewText(transcript) {
                let elapsed = Date().timeIntervalSince(startTime)
                let line: String

                switch format {
                case .text:
                    let ts = formatStreamTimestamp(elapsed)
                    FileHandle.standardError.write(Data("\r\u{1B}[K".utf8))
                    line = "[\(ts)] \(newText)"
                case .jsonl:
                    let jsonObj: [String: Any] = [
                        "time": round(elapsed * 10) / 10,
                        "text": newText,
                        "partial": false,
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

            try await Task.sleep(nanoseconds: 50_000_000) // 50ms polling
        }
    }

    private func log(_ message: String) {
        FileHandle.standardError.write(Data("[scribe] \(message)\n".utf8))
    }
}

private func formatStreamTimestamp(_ seconds: Double) -> String {
    let totalSeconds = Int(seconds)
    let hours = totalSeconds / 3600
    let minutes = (totalSeconds % 3600) / 60
    let secs = totalSeconds % 60
    if hours > 0 {
        return String(format: "%02d:%02d:%02d", hours, minutes, secs)
    }
    return String(format: "%02d:%02d", minutes, secs)
}

enum StreamOutputFormat: String, ExpressibleByArgument, CaseIterable, Sendable {
    case text, jsonl
}
