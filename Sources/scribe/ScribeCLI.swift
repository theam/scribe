import ArgumentParser

@main
struct ScribeCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "scribe",
        abstract: "State-of-the-art local audio transcription with speaker diarization for macOS.",
        version: "0.2.1",
        subcommands: [Transcribe.self, Stream.self, Models.self],
        defaultSubcommand: Transcribe.self
    )
}
