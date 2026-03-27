import Foundation
@preconcurrency import class SpeakerKit.SpeakerKit
import SpeakerKit
import WhisperKit

/// Wraps SpeakerKit to provide speaker diarization.
enum Diarizer {

    static func diarize(
        audioPath: String,
        numSpeakers: Int? = nil,
        verbose: Bool = false
    ) async throws -> DiarizationResult {
        // Load audio as float array
        let audioArray = try AudioProcessor.loadAudioAsFloatArray(fromPath: audioPath)

        // Initialize SpeakerKit with default pyannote config (downloads models if needed)
        let config = PyannoteConfig()
        let manager = SpeakerKitModelManager(config: config)
        try await manager.downloadModels()
        try await manager.loadModels()
        guard let models = manager.models as? PyannoteModels else {
            throw NSError(domain: "Diarizer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to load diarization models"])
        }
        let speakerKit = try SpeakerKit(models: models)

        // Configure diarization options
        let options = PyannoteDiarizationOptions(numberOfSpeakers: numSpeakers)

        // Run diarization
        let result = try await speakerKit.diarize(audioArray: audioArray, options: options)

        return result
    }

    /// Assigns speaker labels to transcription segments using diarization results.
    static func assignSpeakers(
        diarization: DiarizationResult,
        transcription: [TranscriptionResult]
    ) -> [[SpeakerSegment]] {
        return diarization.addSpeakerInfo(to: transcription)
    }
}
