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

    /// Unified entry point for the Nemotron pipeline. Both file and mic modes
    /// flow through the same drain loop — the only difference is the audio source.
    private func runNemotronWithEngine(_ engine: any StreamingAsrEngine) async throws {
        let startTime = Date()
        let outputFile = openOutputFile()

        // Shared queue between the audio source (file or mic) and the drain loop.
        // AsyncStream.Continuation.yield() is non-async and audio-thread-safe — this is
        // FluidAudio's own pattern (see SlidingWindowAsrManager:63-65, 182-184).
        let (stream, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream()

        // Start the source. Mic mode keeps the AVAudioEngine and DispatchSourceSignal
        // alive in `micResources` for the lifetime of the stream.
        var micResources: NemotronMicResources? = nil
        if let path = audioFile {
            try await feedNemotronFromFile(path: path, continuation: continuation)
        } else {
            micResources = try startMicSource(continuation: continuation)
        }

        // Drain loop runs until the continuation finishes (file end or SIGINT).
        try await runNemotronDrainLoop(
            engine: engine,
            stream: stream,
            startTime: startTime,
            outputFile: outputFile
        )

        // Cleanup mic resources after the drain loop has flushed via finish().
        if let res = micResources {
            res.audioEngine.stop()
            res.audioEngine.inputNode.removeTap(onBus: 0)
            res.signalSource.cancel()
        }

        outputFile?.closeFile()
    }

    /// Drain the audio queue, feed the engine, poll the partial transcript, and emit
    /// deltas. Used for both mic and file modes — the queue is the only difference.
    private func runNemotronDrainLoop(
        engine: any StreamingAsrEngine,
        stream: AsyncStream<AVAudioPCMBuffer>,
        startTime: Date,
        outputFile: FileHandle?
    ) async throws {
        let state = StreamState()

        for await buffer in stream {
            nonisolated(unsafe) let buf = buffer
            try await engine.appendAudio(buf)
            try await engine.processBufferedAudio()

            // Live preview to stderr (text format only) — overwritten on each tick.
            let current = await engine.getPartialTranscript()
            emitLivePreview(current, startTime: startTime)

            // Delta emission to stdout (and output file).
            if let newText = await state.getNewText(current) {
                emitDelta(newText, startTime: startTime, outputFile: outputFile)
            }
        }

        // Stream ended — flush any tail audio via finish() and emit remaining content.
        // This is the bug fix from the previous phase: finish() may decode tokens that
        // never reached the per-chunk path, so we always run its result through the
        // delta logic to capture them.
        let finalText = try await engine.finish()
        if let newText = await state.getNewText(finalText) {
            emitDelta(newText, startTime: startTime, outputFile: outputFile)
        }

        // Newline so the final stderr preview line doesn't run into the next prompt.
        if format == .text {
            FileHandle.standardError.write(Data("\n".utf8))
        }
    }

    /// File source adapter — reads the whole file via AudioConverter (16kHz mono Float32),
    /// pushes it as a single buffer, and finishes the continuation so the drain loop exits.
    private func feedNemotronFromFile(
        path: String,
        continuation: AsyncStream<AVAudioPCMBuffer>.Continuation
    ) async throws {
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

        guard let buffer = makeMonoFloat32Buffer(from: samples) else {
            throw ValidationError("Failed to allocate audio buffer")
        }
        continuation.yield(buffer)
        continuation.finish()
    }

    /// Mic source adapter — installs the input tap, pre-resamples each tap buffer to
    /// 16kHz mono Float32 (matching the file path), yields it onto the shared stream,
    /// and wires SIGINT to a DispatchSourceSignal that finishes the continuation
    /// (replacing the brutal Darwin.exit(0) that used to drop tail audio).
    private func startMicSource(
        continuation: AsyncStream<AVAudioPCMBuffer>.Continuation
    ) throws -> NemotronMicResources {
        let audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        if verbose {
            log("Microphone: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount) ch")
        }

        // AudioConverter is a final class with only immutable state — safe to call
        // resampleBuffer() from the audio render thread.
        nonisolated(unsafe) let converter = AudioConverter()
        nonisolated(unsafe) let cont = continuation

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            do {
                let samples = try converter.resampleBuffer(buffer)
                if let outBuffer = makeMonoFloat32Buffer(from: samples) {
                    cont.yield(outBuffer)
                }
            } catch {
                // Drop the buffer on resample error rather than crash the audio thread.
            }
        }

        try audioEngine.start()
        log("Listening (English, low latency)... press Ctrl+C to stop")

        // Graceful shutdown via DispatchSourceSignal — replaces Darwin.exit(0).
        // The handler runs on a regular dispatch queue (not the signal context), so
        // it's safe to call continuation.finish() from here.
        signal(SIGINT, SIG_IGN)
        let sigSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        sigSource.setEventHandler {
            FileHandle.standardError.write(Data("\n[scribe] Stopping...\n".utf8))
            cont.finish()
        }
        sigSource.resume()

        return NemotronMicResources(audioEngine: audioEngine, signalSource: sigSource)
    }

    // MARK: - Nemotron emission helpers

    /// Live preview line on stderr (text format only). Overwrites itself with `\r`.
    private func emitLivePreview(_ fullTranscript: String, startTime: Date) {
        guard format == .text else { return }
        let preview = String(fullTranscript.trimmingCharacters(in: .whitespaces).suffix(100))
        guard !preview.isEmpty else { return }
        let elapsed = Date().timeIntervalSince(startTime)
        let ts = formatStreamTimestamp(elapsed)
        FileHandle.standardError.write(Data("\r\u{1B}[K[\(ts)] \(preview)".utf8))
    }

    /// Emit a confirmed delta to stdout (and output file). Format-aware.
    private func emitDelta(_ newText: String, startTime: Date, outputFile: FileHandle?) {
        let elapsed = Date().timeIntervalSince(startTime)
        let line: String

        switch format {
        case .text:
            // Clear the live preview line before writing the confirmed line on stdout.
            FileHandle.standardError.write(Data("\r\u{1B}[K".utf8))
            let ts = formatStreamTimestamp(elapsed)
            line = "[\(ts)] \(newText)"
        case .jsonl:
            let jsonObj: [String: Any] = [
                "time": round(elapsed * 10) / 10,
                "text": newText,
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

/// Resources held alive by mic mode for the duration of a stream session.
/// The AVAudioEngine and DispatchSourceSignal must outlive the drain loop —
/// dropping them would stop audio capture and cancel the signal handler.
private struct NemotronMicResources {
    let audioEngine: AVAudioEngine
    let signalSource: any DispatchSourceSignal
}

/// Wrap a `[Float]` of 16kHz mono samples into a fresh AVAudioPCMBuffer.
/// Used by both the file source (whole-file buffer) and the mic source
/// (per-tap-buffer wrapping after resampling).
private func makeMonoFloat32Buffer(from samples: [Float]) -> AVAudioPCMBuffer? {
    guard let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16000,
        channels: 1,
        interleaved: false
    ) else {
        return nil
    }
    let frameCount = AVAudioFrameCount(samples.count)
    guard frameCount > 0,
          let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        return nil
    }
    buffer.frameLength = frameCount
    if let channelData = buffer.floatChannelData {
        samples.withUnsafeBufferPointer { src in
            memcpy(channelData[0], src.baseAddress!, samples.count * MemoryLayout<Float>.stride)
        }
    }
    return buffer
}

enum StreamOutputFormat: String, ExpressibleByArgument, CaseIterable, Sendable {
    case text, jsonl
}

enum StreamEngine: String, ExpressibleByArgument, CaseIterable, Sendable {
    case `default`
    case nemotron
}
