import ArgumentParser
import FluidAudio
import Foundation

struct Models: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Manage transcription models.",
        subcommands: [ListModels.self, DownloadModel.self],
        defaultSubcommand: ListModels.self
    )
}

struct ListModels: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List available models."
    )

    func run() async throws {
        print("Available models:\n")
        print("  ASR:")
        print("    parakeet-v3    Parakeet TDT v3 (25 European languages, ~600MB)")
        print("    parakeet-v2    Parakeet TDT v2 (English only, ~600MB)")
        print()
        print("  Diarization:")
        print("    pyannote       Pyannote community-1 (CoreML, ~50MB)")
        print()
        print("Models are downloaded automatically on first use.")
        print("Cache: ~/Library/Application Support/FluidAudio/Models/")
    }
}

struct DownloadModel: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Pre-download models for offline use."
    )

    @Argument(help: "Model to download: asr, diarization, or all.")
    var model: String = "all"

    func run() async throws {
        if model == "asr" || model == "all" {
            print("Downloading ASR model (Parakeet TDT v3)...")
            _ = try await AsrModels.downloadAndLoad(version: .v3)
            print("ASR model ready.")
        }

        if model == "diarization" || model == "all" {
            print("Downloading diarization model...")
            let config = OfflineDiarizerConfig()
            let manager = OfflineDiarizerManager(config: config)
            try await manager.prepareModels()
            print("Diarization model ready.")
        }

        if model != "asr" && model != "diarization" && model != "all" {
            throw ValidationError("Unknown model '\(model)'. Use: asr, diarization, or all.")
        }

        print("Done. Models cached in ~/Library/Application Support/FluidAudio/Models/")
    }
}
