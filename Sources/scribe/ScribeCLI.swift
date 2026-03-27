import ArgumentParser

@main
struct ScribeCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "scribe",
        abstract: "State-of-the-art local audio transcription with speaker diarization for macOS.",
        version: "0.1.0",
        subcommands: [Transcribe.self, Models.self],
        defaultSubcommand: Transcribe.self
    )
}
