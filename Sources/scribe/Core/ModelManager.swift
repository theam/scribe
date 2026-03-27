import ArgumentParser
import Foundation
import WhisperKit

/// Manages downloading, caching, and listing WhisperKit models.
enum ModelManager {

    struct ModelInfo {
        let name: String
        let size: String
    }

    static let availableModels: [ModelInfo] = [
        ModelInfo(name: "tiny", size: "~75MB"),
        ModelInfo(name: "base", size: "~142MB"),
        ModelInfo(name: "small", size: "~466MB"),
        ModelInfo(name: "medium", size: "~1.5GB"),
        ModelInfo(name: "large-v3", size: "~3.1GB"),
        ModelInfo(name: "large-v3-turbo", size: "~632MB"),
        ModelInfo(name: "distil-large-v3", size: "~756MB"),
    ]

    static let defaultModel = "large-v3-turbo"

    /// Returns the names of models that have been downloaded locally.
    static func downloadedModels() -> [String] {
        let modelsDir = modelsDirectory()
        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: modelsDir.path) else {
            return []
        }
        return contents.filter { !$0.hasPrefix(".") }
    }

    /// Downloads a model and returns the local path.
    static func download(model: String) async throws -> String {
        let folder = try await WhisperKit.download(variant: model, from: "argmaxinc/whisperkit-coreml")
        return folder.path
    }

    /// Removes a downloaded model.
    static func remove(model: String) throws {
        let modelsDir = modelsDirectory()
        let modelPath = modelsDir.appendingPathComponent(model)
        if FileManager.default.fileExists(atPath: modelPath.path) {
            try FileManager.default.removeItem(at: modelPath)
        } else {
            throw ValidationError("Model '\(model)' is not downloaded.")
        }
    }

    private static func modelsDirectory() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("huggingface/models/argmaxinc/whisperkit-coreml")
    }
}
