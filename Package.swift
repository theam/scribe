// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "scribe",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "scribe", targets: ["scribe"])
    ],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.12.4"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "scribe",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
