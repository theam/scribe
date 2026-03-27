import ArgumentParser
import Foundation
import WhisperKit

struct Models: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Manage transcription models.",
        subcommands: [ListModels.self, DownloadModel.self, RemoveModel.self],
        defaultSubcommand: ListModels.self
    )
}

struct ListModels: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List available and downloaded models."
    )

    func run() async throws {
        let localModels = ModelManager.downloadedModels()
        let availableModels = ModelManager.availableModels

        print("Available models:\n")
        for model in availableModels {
            let downloaded = localModels.contains(model.name)
            let marker = downloaded ? "[downloaded]" : ""
            print("  \(model.name)\t\(model.size)\t\(marker)")
        }

        if localModels.isEmpty {
            print("\nNo models downloaded. Run 'scribe models download large-v3-turbo' to get started.")
        }
    }
}

struct DownloadModel: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Download a model."
    )

    @Argument(help: "Model name to download (e.g., large-v3-turbo).")
    var name: String

    func run() async throws {
        print("Downloading model '\(name)'...")
        let path = try await ModelManager.download(model: name)
        print("Downloaded to: \(path)")
    }
}

struct RemoveModel: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "remove",
        abstract: "Remove a downloaded model."
    )

    @Argument(help: "Model name to remove.")
    var name: String

    func run() async throws {
        try ModelManager.remove(model: name)
        print("Removed model '\(name)'.")
    }
}
