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

    @Option(name: .long, help: "Read audio from a WAV/audio file instead of microphone (for testing/eval). Exits when file is consumed.")
    var audioFile: String?

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
        log("Initializing streaming ASR (Nemotron 1120ms, English-only)...")
        log("Downloading model if needed (first run only, ~600MB)...")

        let engine = StreamingAsrEngineFactory.create(.nemotron1120ms)

        do {
            try await engine.loadModels()
        } catch {
            log("Model loading failed: \(error.localizedDescription)")
            log("Cleaning cache and retrying download...")

            let cacheDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                .appendingPathComponent("FluidAudio/Models/nemotron-streaming")
            try? FileManager.default.removeItem(at: cacheDir)

            let freshEngine = StreamingAsrEngineFactory.create(.nemotron1120ms)
            try await freshEngine.loadModels()
            log("Retry successful.")
            try await runNemotronWithEngine(freshEngine)
            return
        }

        log("Models loaded.")
        try await runNemotronWithEngine(engine)
    }

    private func runNemotronWithEngine(_ engine: any StreamingAsrEngine) async throws {
        let startTime = Date()
        let outputFile = openOutputFile()

        if let path = audioFile {
            // File mode: do NOT set up the per-chunk callback. Use finish() as the
            // authoritative transcript source. The callback path drops content because
            // (a) finish() may decode tokens after the last chunk callback, and
            // (b) Task { } dispatched from the callback may not run before exit.
            try await runNemotronFromFile(engine: engine, path: path, startTime: startTime, outputFile: outputFile)
        } else {
            // Mic mode: live preview via per-chunk callback (acceptable streaming UX).
            await setupNemotronCallback(engine: engine, startTime: startTime, outputFile: outputFile)
            try await runNemotronFromMic(engine: engine)
        }
    }

    private func runNemotronFromMic(engine: any StreamingAsrEngine) async throws {
        let audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        if verbose {
            log("Microphone: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount) ch")
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

    /// Feed a pre-recorded audio file into the streaming engine.
    /// Reads the whole file via FluidAudio's AudioConverter (16kHz mono Float32),
    /// chunks it to mimic the live mic tap, calls finish() for the authoritative
    /// transcript, and emits it once. No per-chunk callback (see runNemotronWithEngine).
    private func runNemotronFromFile(engine: any StreamingAsrEngine, path: String, startTime: Date, outputFile: FileHandle?) async throws {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("Audio file not found: \(path)")
        }

        if verbose { log("Loading audio file: \(path)") }
        let converter = AudioConverter()
        let samples = try converter.resampleAudioFile(url)
        let duration = Double(samples.count) / 16000.0
        if verbose {
            log(String(format: "Loaded %.0fs (%.1f min), %d samples", duration, duration / 60, samples.count))
        }
        log("Streaming from file (no microphone)...")

        // Match the official FluidAudio reference pattern (NemotronTranscribe.swift):
        // load the WHOLE file into a single AVAudioPCMBuffer and feed it as one
        // appendAudio call. The engine handles internal chunking. This avoids any
        // boundary issues from feeding the engine many tiny buffers in a loop.
        guard let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        ) else {
            throw ValidationError("Failed to create audio format")
        }

        let frameCount = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
            throw ValidationError("Failed to allocate audio buffer")
        }
        buffer.frameLength = frameCount
        if let channelData = buffer.floatChannelData {
            samples.withUnsafeBufferPointer { src in
                memcpy(channelData[0], src.baseAddress!, samples.count * MemoryLayout<Float>.stride)
            }
        }
        nonisolated(unsafe) let buf = buffer
        try await engine.appendAudio(buf)
        try await engine.processBufferedAudio()

        // Authoritative final transcript: finish() pads any trailing partial chunk,
        // returns the full decoded text, and clears internal state. We use this as
        // the source of truth (not the per-chunk callback, which drops content).
        let finalText = try await engine.finish()
        if verbose { log(String(format: "Finalized. Final transcript: %d chars", finalText.count)) }

        // Emit the complete transcript as a single result.
        emitFinalTranscript(finalText, startTime: startTime, outputFile: outputFile)
    }

    /// Emit the final (authoritative) transcript from finish() as one record.
    /// In file mode this is the only output that goes to stdout — no per-chunk deltas.
    private func emitFinalTranscript(_ text: String, startTime: Date, outputFile: FileHandle?) {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }

        let elapsed = Date().timeIntervalSince(startTime)
        let line: String

        switch format {
        case .text:
            let ts = formatStreamTimestamp(elapsed)
            line = "[\(ts)] \(trimmed)"
        case .jsonl:
            let jsonObj: [String: Any] = [
                "time": round(elapsed * 10) / 10,
                "text": trimmed,
            ]
            guard let data = try? JSONSerialization.data(withJSONObject: jsonObj, options: [.sortedKeys]),
                  let jsonStr = String(data: data, encoding: .utf8) else {
                return
            }
            line = jsonStr
        }

        print(line)
        fflush(stdout)
        outputFile?.write(Data((line + "\n").utf8))
    }

    /// Set up the partial-transcript callback used by both mic and file modes.
    private func setupNemotronCallback(engine: any StreamingAsrEngine, startTime: Date, outputFile: FileHandle?) async {
        let state = StreamState()
        let fmt = format

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
    }

    private func openOutputFile() -> FileHandle? {
        guard let outputPath = output else { return nil }
        FileManager.default.createFile(atPath: outputPath, contents: nil)
        return FileHandle(forWritingAtPath: outputPath)
    }

    // MARK: - SlidingWindow Engine (Multilingual, default)

    private func runSlidingWindow() async throws {
        if audioFile != nil {
            throw ValidationError("--audio-file is not yet supported with the multilingual engine. Use --engine nemotron for now.")
        }

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
