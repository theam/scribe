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

    @Option(name: .long, help: "Streaming engine: default (multilingual, higher latency) or nemotron (English-only, low latency ~560ms).")
    var engine: StreamEngine = .default

    @Option(name: .shortAndLong, help: "Also save output to file.")
    var output: String?

    @Flag(name: .long, help: "Show status information.")
    var verbose: Bool = false

    func run() async throws {
        switch engine {
        case .nemotron:
            try await runNemotron()
        case .default:
            try await runSlidingWindow()
        }
    }

    // MARK: - Nemotron Engine (English-only, low latency)

    private func runNemotron() async throws {
        if verbose { log("Initializing streaming ASR (Nemotron 560ms, English-only)...") }

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

        setupSignalHandler()

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            nonisolated(unsafe) let buf = buffer
            do { try engine.appendAudio(buf) } catch {}
        }

        try audioEngine.start()
        log("Listening (English, low latency)... press Ctrl+C to stop")

        while true {
            try await engine.processBufferedAudio()

            let transcript = await engine.getPartialTranscript()
            if let newText = await state.getNewText(transcript) {
                let elapsed = Date().timeIntervalSince(startTime)
                emitLine(text: newText, elapsed: elapsed, partial: false, outputFile: outputFile)
            }

            try await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    // MARK: - SlidingWindow Engine (Multilingual, default)

    private func runSlidingWindow() async throws {
        if verbose { log("Initializing streaming ASR (Parakeet TDT v3, multilingual)...") }

        let streamManager = SlidingWindowAsrManager(config: .streaming)
        try await streamManager.start(source: .microphone)

        if verbose { log("Models loaded.") }

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

        setupSignalHandler()

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            nonisolated(unsafe) let buf = buffer
            streamManager.streamAudio(buf)
        }

        try audioEngine.start()
        log("Listening (multilingual)... press Ctrl+C to stop")

        let updates = await streamManager.transcriptionUpdates
        for await update in updates {
            let text = update.text.trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { continue }

            let elapsed = Date().timeIntervalSince(startTime)

            if update.isConfirmed {
                guard text != lastConfirmedText else { continue }
                lastConfirmedText = text
                lastVolatileText = ""
                emitLine(text: text, elapsed: elapsed, partial: false, outputFile: outputFile)
            } else {
                guard text != lastVolatileText else { continue }
                lastVolatileText = text

                switch format {
                case .text:
                    let ts = formatStreamTimestamp(elapsed)
                    let preview = String(text.suffix(100))
                    FileHandle.standardError.write(Data("\r\u{1B}[K[\(ts)] \(preview)".utf8))
                case .jsonl:
                    emitLine(text: text, elapsed: elapsed, partial: true, outputFile: nil)
                }
            }
        }

        audioEngine.stop()
        inputNode.removeTap(onBus: 0)
        _ = try await streamManager.finish()
    }

    // MARK: - Shared Helpers

    private func emitLine(text: String, elapsed: Double, partial: Bool, outputFile: FileHandle?) {
        let line: String

        switch format {
        case .text:
            let ts = formatStreamTimestamp(elapsed)
            FileHandle.standardError.write(Data("\r\u{1B}[K".utf8))
            line = "[\(ts)] \(text)"
        case .jsonl:
            let jsonObj: [String: Any] = [
                "time": round(elapsed * 10) / 10,
                "text": text,
                "partial": partial,
            ]
            if let data = try? JSONSerialization.data(withJSONObject: jsonObj, options: [.sortedKeys]),
               let jsonStr = String(data: data, encoding: .utf8) {
                line = jsonStr
            } else {
                return
            }
        }

        print(line)
        fflush(stdout)

        if let file = outputFile {
            file.write(Data((line + "\n").utf8))
        }
    }

    private func setupSignalHandler() {
        signal(SIGINT) { _ in
            FileHandle.standardError.write(Data("\n".utf8))
            Darwin.exit(0)
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

enum StreamEngine: String, ExpressibleByArgument, CaseIterable, Sendable {
    case `default`
    case nemotron
}
