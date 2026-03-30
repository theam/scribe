import ArgumentParser
import AVFoundation
import FluidAudio
import Foundation

/// Thread-safe state for tracking incremental transcript output.
private actor StreamState {
    private var lastFullTranscript = ""
    private var printedLength = 0

    /// Given the full accumulated transcript, return only the new portion.
    /// Handles model revisions by finding the longest common prefix.
    func getNewText(_ fullTranscript: String) -> String? {
        let text = fullTranscript.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return nil }
        guard text != lastFullTranscript else { return nil }

        lastFullTranscript = text

        // Find how much of the text we've already printed
        if text.count > printedLength {
            let newPortion = String(text.dropFirst(printedLength)).trimmingCharacters(in: .whitespaces)
            if !newPortion.isEmpty {
                printedLength = text.count
                return newPortion
            }
        }

        return nil
    }

    /// Get the live preview (last N chars of full transcript).
    func getPreview(_ fullTranscript: String, maxLen: Int = 100) -> String? {
        let text = fullTranscript.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return nil }
        guard text != lastFullTranscript else { return nil }
        return String(text.suffix(maxLen))
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
        log("Initializing streaming ASR (Nemotron 560ms, English-only)...")
        log("Downloading model if needed (first run only, ~600MB)...")

        let engine = StreamingAsrEngineFactory.create(.nemotron560ms)

        do {
            try await engine.loadModels()
        } catch {
            log("Model loading failed: \(error.localizedDescription)")
            log("Cleaning cache and retrying download...")

            let cacheDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                .appendingPathComponent("FluidAudio/Models/nemotron-streaming")
            try? FileManager.default.removeItem(at: cacheDir)

            let freshEngine = StreamingAsrEngineFactory.create(.nemotron560ms)
            try await freshEngine.loadModels()
            log("Retry successful.")
            try await runNemotronWithEngine(freshEngine)
            return
        }

        log("Models loaded.")
        try await runNemotronWithEngine(engine)
    }

    private func runNemotronWithEngine(_ engine: any StreamingAsrEngine) async throws {
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

        // Capture file handle for sendable closure
        nonisolated(unsafe) let outFile = outputFile

        // Partial callback — fires after each 560ms chunk with the full accumulated transcript.
        // We diff to find new text and emit only the delta.
        await engine.setPartialTranscriptCallback { fullTranscript in
            Task {
                let elapsed = Date().timeIntervalSince(startTime)

                switch fmt {
                case .text:
                    // Show live preview on stderr (ephemeral, overwritten)
                    let preview = String(fullTranscript.trimmingCharacters(in: .whitespaces).suffix(100))
                    if !preview.isEmpty {
                        let ts = formatStreamTimestamp(elapsed)
                        FileHandle.standardError.write(Data("\r\u{1B}[K[\(ts)] \(preview)".utf8))
                    }
                case .jsonl:
                    break
                }

                // Emit new portion to stdout
                if let newText = await state.getNewText(fullTranscript) {
                    let ts = formatStreamTimestamp(elapsed)

                    let line: String
                    switch fmt {
                    case .text:
                        FileHandle.standardError.write(Data("\r\u{1B}[K".utf8))
                        line = "[\(ts)] \(newText)"
                    case .jsonl:
                        let jsonObj: [String: Any] = [
                            "time": round(elapsed * 10) / 10,
                            "text": newText,
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

                    if let file = outFile {
                        file.write(Data((line + "\n").utf8))
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

        // Processing loop — drives the engine to process buffered audio
        while true {
            try await engine.processBufferedAudio()
            try await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }
    }

    // MARK: - SlidingWindow Engine (Multilingual, default)

    private func runSlidingWindow() async throws {
        log("Initializing streaming ASR (Parakeet TDT v3, multilingual)...")
        log("Downloading model if needed (first run only, ~600MB)...")

        let streamManager = SlidingWindowAsrManager(config: .streaming)
        try await streamManager.start(source: .microphone)

        log("Models loaded.")

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
                emitLine(text: text, elapsed: elapsed, outputFile: outputFile)
            } else {
                guard text != lastVolatileText else { continue }
                lastVolatileText = text

                switch format {
                case .text:
                    let ts = formatStreamTimestamp(elapsed)
                    let preview = String(text.suffix(100))
                    FileHandle.standardError.write(Data("\r\u{1B}[K[\(ts)] \(preview)".utf8))
                case .jsonl:
                    break
                }
            }
        }

        audioEngine.stop()
        inputNode.removeTap(onBus: 0)
        _ = try await streamManager.finish()
    }

    // MARK: - Helpers

    private func emitLine(text: String, elapsed: Double, outputFile: FileHandle?) {
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
