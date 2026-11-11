import SwiftUI
import UniformTypeIdentifiers

@main
struct SDMetalApp: App {
    @StateObject private var engine = SDEngine()
    @StateObject private var models = ModelStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(engine)
                .environmentObject(models)
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var engine: SDEngine
    @EnvironmentObject var models: ModelStore

    @State private var loadConfig = LoadConfig()
    @State private var genConfig = GenConfig()
    @State private var showPicker = false
    @State private var pickingKey = ""

    private var canLoad: Bool {
        return !models.path("diffusion").isEmpty
            && !models.path("llm").isEmpty
            && !models.path("vae").isEmpty
    }

    var body: some View {
        NavigationStack {
            Form {
                modelsSection
                engineSection
                generateSection
                statusSection
                resultSection
                logSection
                previousLogSection
            }
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled(true)
            .navigationTitle("SD Metal")
            .fileImporter(
                isPresented: $showPicker,
                allowedContentTypes: [.item],
                allowsMultipleSelection: false
            ) { result in
                if case .success(let urls) = result, let url = urls.first {
                    models.choose(url, for: pickingKey)
                }
            }
        }
    }

    // MARK: Sections

    private var modelsSection: some View {
        Section("Model files") {
            ForEach(ModelStore.slots) { slot in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(slot.title).font(.subheadline)
                        Text(models.urls[slot.id]?.lastPathComponent ?? (slot.required ? "not chosen" : "optional"))
                            .font(.caption)
                            .foregroundColor(.secondary)
                        if let note = models.notes[slot.id], !note.isEmpty {
                            Text(note).font(.caption2).foregroundColor(.orange)
                        }
                    }
                    Spacer()
                    if models.urls[slot.id] != nil {
                        Button("Clear") {
                            models.clear(slot.id)
                        }
                        .buttonStyle(.borderless)
                    }
                    Button("Choose") {
                        pickingKey = slot.id
                        showPicker = true
                    }
                    .buttonStyle(.borderless)
                }
            }
        }
    }

    private var engineSection: some View {
        Section("Engine settings (applied on Load)") {
            TextField("backend (empty = automatic)", text: $loadConfig.backend)
            TextField("params backend (empty = automatic)", text: $loadConfig.paramsBackend)
            TextField("max memory budget in GiB (empty = automatic)", text: $loadConfig.maxVram)
            TextField("model args", text: $loadConfig.modelArgs)
            Picker("flash attention", selection: $loadConfig.flashAttn) {
                Text("auto").tag("auto")
                Text("on").tag("on")
                Text("off").tag("off")
            }
            TextField("conv direct mode (empty = default)", text: $loadConfig.convDirect)
            Toggle("memory map files", isOn: $loadConfig.mmap)
            Toggle("auto fit memory", isOn: $loadConfig.autoFit)
            Toggle("disable segmented compute", isOn: $loadConfig.noSegment)
            Toggle("disable prefetch", isOn: $loadConfig.noPrefetch)
            Toggle("load all weights at load time", isOn: $loadConfig.eager)
            HStack {
                Text("linear scale")
                Spacer()
                TextField("0", value: $loadConfig.linearScale, format: .number)
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 120)
            }
            HStack {
                Text("attention scale")
                Spacer()
                TextField("0", value: $loadConfig.attnScale, format: .number)
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 120)
            }
            Stepper(
                "threads: " + (loadConfig.threads == 0 ? "automatic" : String(loadConfig.threads)),
                value: $loadConfig.threads,
                in: 0...12
            )
            HStack {
                Button("Load") {
                    engine.load(
                        paths: ModelPaths(
                            diffusion: models.path("diffusion"),
                            llm: models.path("llm"),
                            vision: models.path("vision"),
                            vae: models.path("vae")
                        ),
                        config: loadConfig
                    )
                }
                .buttonStyle(.borderless)
                .disabled(engine.busy || !canLoad)
                Spacer()
                Button("Unload") {
                    engine.unload()
                }
                .buttonStyle(.borderless)
                .disabled(engine.busy || !engine.loaded)
            }
        }
    }

    private var generateSection: some View {
        Section("Generate") {
            TextField("prompt", text: $genConfig.prompt, axis: .vertical)
                .lineLimit(2...6)
            TextField("negative prompt", text: $genConfig.negative, axis: .vertical)
                .lineLimit(1...3)
            HStack {
                Text("width")
                Spacer()
                TextField("512", value: $genConfig.width, format: .number)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }
            HStack {
                Text("height")
                Spacer()
                TextField("512", value: $genConfig.height, format: .number)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }
            HStack {
                Text("steps")
                Spacer()
                TextField("8", value: $genConfig.steps, format: .number)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }
            HStack {
                Text("guidance scale")
                Spacer()
                TextField("1.0", value: $genConfig.cfg, format: .number)
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }
            HStack {
                Text("seed")
                Spacer()
                TextField("42", value: $genConfig.seed, format: .number)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 140)
            }
            TextField("custom sigmas, comma separated (optional)", text: $genConfig.sigmas)
            TextField("extra sample args (optional)", text: $genConfig.extraSampleArgs)
            HStack {
                Button("Generate") {
                    engine.generate(config: genConfig)
                }
                .buttonStyle(.borderless)
                .disabled(engine.busy || !engine.loaded)
                Spacer()
                Button("Profile") {
                    engine.generate(config: genConfig, profile: true)
                }
                .buttonStyle(.borderless)
                .disabled(engine.busy || !engine.loaded)
                Spacer()
                Button("Cancel") {
                    engine.cancel()
                }
                .buttonStyle(.borderless)
                .disabled(!engine.busy)
            }
        }
    }

    private var statusSection: some View {
        Section("Status") {
            Text(engine.status)
            if !engine.progressText.isEmpty {
                Text(engine.progressText)
                    .font(.caption)
            }
            Text("free memory: \(engine.freeMemoryMB) MB, thermal: \(engine.thermal)")
                .font(.caption)
            DisclosureGroup("Devices") {
                Text(engine.devices)
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
            }
        }
    }

    @ViewBuilder
    private var resultSection: some View {
        if let image = engine.image {
            Section("Result") {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                ShareLink(
                    item: Image(uiImage: image),
                    preview: SharePreview("Generated image", image: Image(uiImage: image))
                )
            }
        }
    }

    private var logSection: some View {
        Section("Log (last 60 lines, also saved to run.log)") {
            Text(engine.logLines.suffix(60).joined(separator: "\n"))
                .font(.system(size: 10, design: .monospaced))
                .textSelection(.enabled)
            ShareLink(item: engine.logFileURL)
        }
    }

    @ViewBuilder
    private var previousLogSection: some View {
        if !engine.previousLog.isEmpty {
            Section("Previous run, end of log") {
                Text(engine.previousLog)
                    .font(.system(size: 10, design: .monospaced))
                    .textSelection(.enabled)
            }
        }
    }
}
