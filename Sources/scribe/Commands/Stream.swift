import ArgumentParser
import AVFoundation
import FluidAudio
import Foundation

struct Stream: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Stream live audio transcription from microphone or system audio."
    )

    @Flag(name: .long, help: "Capture microphone audio (default if no --source specified).")
    var mic: Bool = false

    @Option(name: .long, help: "Output format: text or jsonl.")
    var format: StreamOutputFormat = .text

    @Option(name: .shortAndLong, help: "Also save output to file.")
    var output: String?

    @Flag(name: .long, help: "Show status information.")
    var verbose: Bool = false

    func run() async throws {
        // For now, mic is the default and only source
        if verbose { log("Initializing streaming ASR (Parakeet)...") }

        let streamManager = SlidingWindowAsrManager(config: .streaming)
        try await streamManager.start(source: .microphone)

        // Set up microphone capture
        let audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        if verbose {
            log("Microphone: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount) ch")
            log("Listening... (press Ctrl+C to stop)")
        }

        // Track state for output
        let startTime = Date()
        var outputFile: FileHandle? = nil
        var utteranceCount = 0

        if let outputPath = output {
            FileManager.default.createFile(atPath: outputPath, contents: nil)
            outputFile = FileHandle(forWritingAtPath: outputPath)
        }

        // Set up signal handler for clean Ctrl+C
        signal(SIGINT) { _ in
            Darwin.exit(0)
        }

        // Install tap on microphone — feed buffers to ASR
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            // nonisolated(unsafe) to cross sendability boundary — buffer is consumed immediately
            nonisolated(unsafe) let buf = buffer
            streamManager.streamAudio(buf)
        }

        try audioEngine.start()

        if !verbose {
            // Print a minimal status line to stderr
            log("Listening... (press Ctrl+C to stop)")
        }

        // Read transcription updates
        let updates = await streamManager.transcriptionUpdates
        for await update in updates {
            // Stream continues until Ctrl+C or stream ends

            let text = update.text.trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { continue }

            utteranceCount += 1
            let elapsed = Date().timeIntervalSince(startTime)
            let speaker = mic ? "You" : "Others"

            let line: String
            switch format {
            case .text:
                let ts = formatTimestamp(elapsed)
                line = "[\(ts)] \(speaker): \(text)"
            case .jsonl:
                let jsonObj: [String: Any] = [
                    "time": round(elapsed * 10) / 10,
                    "speaker": speaker,
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
        let finalText = try await streamManager.finish()

        let totalElapsed = Date().timeIntervalSince(startTime)

        // Print summary to stderr
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

// MARK: - Stream Output Format

enum StreamOutputFormat: String, ExpressibleByArgument, CaseIterable, Sendable {
    case text, jsonl
}
