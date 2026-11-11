import Foundation
import UIKit
import os

// MARK: - Settings structs

struct ModelPaths {
    var diffusion = ""
    var llm = ""
    var vision = ""
    var vae = ""
}

struct LoadConfig {
    var backend = ""
    var paramsBackend = ""
    var maxVram = "6"
    var modelArgs = ""
    var flashAttn = "auto"
    var convDirect = ""
    var threads = 0
    var linearScale = 0.0
    var attnScale = 0.0
    var mmap = false
    var autoFit = true
    var noSegment = false
    var noPrefetch = false
    var eager = false
}

struct GenConfig {
    var prompt = "a red fox sitting in fresh snow, soft morning light"
    var negative = ""
    var width = 512
    var height = 512
    var steps = 8
    var cfg = 1.0
    var seed = 42
    var sigmas = ""
    var extraSampleArgs = ""
}

// MARK: - C callbacks (no captured context; the engine travels in the user pointer)

private let engineLogCallback: eng_log_cb = { _, text, user in
    guard let text = text, let user = user else { return }
    let engine = Unmanaged<SDEngine>.fromOpaque(user).takeUnretainedValue()
    engine.log(String(cString: text))
}

private let engineProgressCallback: eng_progress_cb = { stage, detail, done, total, user in
    guard let user = user else { return }
    let engine = Unmanaged<SDEngine>.fromOpaque(user).takeUnretainedValue()
    let stageText = stage.map { String(cString: $0) } ?? ""
    let detailText = detail.map { String(cString: $0) } ?? ""
    engine.progress(stage: stageText, detail: detailText, done: Int(done), total: Int(total))
}

// MARK: - Engine

final class SDEngine: ObservableObject {
    @Published var logLines: [String] = []
    @Published var previousLog = ""
    @Published var status = "Idle"
    @Published var progressText = ""
    @Published var image: UIImage?
    @Published var freeMemoryMB = 0
    @Published var thermal = ""
    @Published var devices = ""
    @Published var loaded = false
    @Published var busy = false

    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    var logFileURL: URL { documents.appendingPathComponent("run.log") }

    private var logHandle: FileHandle?
    private let fileQueue = DispatchQueue(label: "sdmetal.log")
    private var timer: Timer?
    private let startTime = Date()

    init() {
        rotateLogs()
        eng_sd_set_callbacks(engineLogCallback, engineProgressCallback, Unmanaged.passUnretained(self).toOpaque())
        log("commit \(String(cString: eng_sd_commit())) version \(String(cString: eng_sd_version()))")
        log("system: \(String(cString: eng_sd_system_info()))")
        devices = readDevices()
        log("devices:\n\(devices)")
        updateStats()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
    }

    // MARK: Logging

    private func rotateLogs() {
        let manager = FileManager.default
        let previousURL = documents.appendingPathComponent("previous-run.log")
        if manager.fileExists(atPath: logFileURL.path) {
            if let text = try? String(contentsOf: logFileURL, encoding: .utf8) {
                let tail = text.split(separator: "\n", omittingEmptySubsequences: true).suffix(40)
                previousLog = tail.joined(separator: "\n")
            }
            try? manager.removeItem(at: previousURL)
            try? manager.moveItem(at: logFileURL, to: previousURL)
        }
        manager.createFile(atPath: logFileURL.path, contents: nil)
        logHandle = try? FileHandle(forWritingTo: logFileURL)
    }

    func log(_ message: String) {
        let freeMB = Double(os_proc_available_memory()) / 1_048_576.0
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return }
        let elapsed = Date().timeIntervalSince(startTime)
        let line = String(format: "[+%.1fs %.0f MB free] ", elapsed, freeMB) + trimmed
        fileQueue.sync {
            if let data = (line + "\n").data(using: .utf8) {
                try? self.logHandle?.write(contentsOf: data)
            }
        }
        DispatchQueue.main.async {
            self.logLines.append(line)
            if self.logLines.count > 300 {
                self.logLines.removeFirst(self.logLines.count - 300)
            }
        }
    }

    func progress(stage: String, detail: String, done: Int, total: Int) {
        let text = "\(stage) \(detail) \(done)/\(total)"
        log("progress: " + text)
        DispatchQueue.main.async {
            self.progressText = text
        }
    }

    // MARK: Stats

    private func readDevices() -> String {
        let needed = eng_sd_devices(nil, 0)
        if needed == 0 { return "(none reported)" }
        var buffer = [CChar](repeating: 0, count: needed + 1)
        let capacity = buffer.count
        _ = eng_sd_devices(&buffer, capacity)
        return String(cString: buffer)
    }

    private func updateStats() {
        freeMemoryMB = Int(os_proc_available_memory() / 1_048_576)
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }
    }

    private func tick() {
        updateStats()
        if busy {
            log("tick: thermal \(thermal)")
        }
    }

    // MARK: Load, generate, cancel

    func load(paths: ModelPaths, config: LoadConfig) {
        if busy { return }
        busy = true
        status = "Loading"
        DispatchQueue.global(qos: .userInitiated).async {
            var allocated: [UnsafeMutablePointer<CChar>] = []
            func cstr(_ text: String) -> UnsafePointer<CChar>? {
                if text.isEmpty { return nil }
                guard let copy = strdup(text) else { return nil }
                allocated.append(copy)
                return UnsafePointer(copy)
            }

            var params = eng_sd_load_params()
            params.diffusion_path = cstr(paths.diffusion)
            params.llm_path = cstr(paths.llm)
            params.llm_vision_path = cstr(paths.vision)
            params.vae_path = cstr(paths.vae)
            params.backend = cstr(config.backend)
            params.params_backend = cstr(config.paramsBackend)
            params.max_vram = cstr(config.maxVram)
            params.model_args = cstr(config.modelArgs)
            params.flash_attn_mode = cstr(config.flashAttn)
            params.conv_direct_mode = cstr(config.convDirect)
            params.n_threads = Int32(config.threads)
            params.linear_scale = Float(config.linearScale)
            params.attn_scale = Float(config.attnScale)
            params.enable_mmap = config.mmap
            params.auto_fit = config.autoFit
            params.disable_segmented_compute = config.noSegment
            params.disable_prefetch = config.noPrefetch
            params.eager_load = config.eager

            var errorBuffer = [CChar](repeating: 0, count: 2048)
            let errorCapacity = errorBuffer.count
            let ok = eng_sd_load(&params, &errorBuffer, errorCapacity)
            let message = String(cString: errorBuffer)
            for pointer in allocated { free(pointer) }

            self.log(ok ? "load ok" : "load failed: \(message)")
            DispatchQueue.main.async {
                self.busy = false
                self.loaded = ok
                self.status = ok ? "Loaded" : "Load failed: \(message)"
            }
        }
    }

    func unload() {
        if busy { return }
        eng_sd_unload()
        loaded = false
        status = "Unloaded"
        log("unloaded")
    }

    func generate(config requested: GenConfig, profile: Bool = false) {
        if busy || !loaded { return }
        var config = requested
        if profile {
            config.width = 512
            config.height = 512
            config.steps = 2
        }
        busy = true
        status = profile ? "Profiling (512x512, 2 steps)" : "Generating"
        progressText = ""
        UIApplication.shared.isIdleTimerDisabled = true
        let finalConfig = config
        DispatchQueue.global(qos: .userInitiated).async {
            let config = finalConfig
            var allocated: [UnsafeMutablePointer<CChar>] = []
            func cstr(_ text: String) -> UnsafePointer<CChar>? {
                if text.isEmpty { return nil }
                guard let copy = strdup(text) else { return nil }
                allocated.append(copy)
                return UnsafePointer(copy)
            }

            var params = eng_sd_gen_params()
            params.prompt = cstr(config.prompt)
            params.negative_prompt = cstr(config.negative)
            params.width = Int32(config.width)
            params.height = Int32(config.height)
            params.steps = Int32(config.steps)
            params.cfg_scale = Float(config.cfg)
            params.seed = Int64(config.seed)
            params.sample_method = -1
            params.scheduler = -1
            params.sigmas_csv = cstr(config.sigmas)
            params.extra_sample_args = cstr(config.extraSampleArgs)

            var pixels: UnsafeMutablePointer<UInt8>? = nil
            var width: Int32 = 0
            var height: Int32 = 0
            var channels: Int32 = 0
            var info: UnsafeMutablePointer<CChar>? = nil
            var errorBuffer = [CChar](repeating: 0, count: 2048)
            let errorCapacity = errorBuffer.count

            if profile { eng_sd_profile_start() }
            let started = Date()
            let ok = eng_sd_generate(&params, &pixels, &width, &height, &channels, &info, &errorBuffer, errorCapacity)
            let elapsed = Date().timeIntervalSince(started)
            if profile {
                var report = [CChar](repeating: 0, count: 16384)
                let reportCapacity = report.count
                _ = eng_sd_profile_report(&report, reportCapacity)
                for line in String(cString: report).split(separator: "\n") {
                    self.log("PROFILE " + String(line))
                }
            }
            let message = String(cString: errorBuffer)
            for pointer in allocated { free(pointer) }

            var result: UIImage? = nil
            var savedName = ""
            if ok, let pixels = pixels {
                result = SDEngine.makeImage(pixels: pixels, width: Int(width), height: Int(height), channels: Int(channels))
                if let png = result?.pngData() {
                    let formatter = DateFormatter()
                    formatter.dateFormat = "yyyyMMdd-HHmmss"
                    savedName = "out-" + formatter.string(from: Date()) + ".png"
                    try? png.write(to: self.documents.appendingPathComponent(savedName))
                }
            }
            if let pixels = pixels { eng_free(pixels) }
            if let info = info {
                self.log("generation info: " + String(cString: info))
                eng_free(info)
            }

            var summary = "Failed: \(message)"
            if ok {
                summary = String(format: "Done in %.1f s", elapsed)
                if !savedName.isEmpty {
                    summary += ", saved " + savedName
                }
            }
            self.log(summary)
            let finalSummary = summary
            let finalImage = result
            DispatchQueue.main.async {
                UIApplication.shared.isIdleTimerDisabled = false
                self.busy = false
                self.status = finalSummary
                if let finalImage = finalImage { self.image = finalImage }
            }
        }
    }

    func cancel() {
        eng_sd_cancel()
        log("cancel requested")
    }

    // MARK: Image conversion

    static func makeImage(pixels: UnsafeMutablePointer<UInt8>, width: Int, height: Int, channels: Int) -> UIImage? {
        if channels != 3 && channels != 4 { return nil }
        let data = Data(bytes: pixels, count: width * height * channels)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        let alpha: CGImageAlphaInfo = channels == 4 ? .noneSkipLast : .none
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8 * channels,
            bytesPerRow: width * channels,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: alpha.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}

// MARK: - Model file store

final class ModelStore: ObservableObject {
    struct Slot: Identifiable {
        let id: String
        let title: String
        let required: Bool
    }

    static let slots: [Slot] = [
        Slot(id: "diffusion", title: "Diffusion model (.gguf)", required: true),
        Slot(id: "llm", title: "Text encoder (.gguf)", required: true),
        Slot(id: "vae", title: "VAE (.safetensors)", required: true),
        Slot(id: "vision", title: "Vision projector (.gguf, editing only)", required: false)
    ]

    @Published var urls: [String: URL] = [:]
    @Published var notes: [String: String] = [:]

    private let defaults = UserDefaults.standard

    init() {
        for slot in ModelStore.slots {
            restore(slot.id)
        }
    }

    func choose(_ url: URL, for key: String) {
        let scoped = url.startAccessingSecurityScopedResource()
        urls[key] = url
        do {
            let data = try url.bookmarkData()
            defaults.set(data, forKey: "bookmark." + key)
            notes[key] = scoped ? "" : "no scoped access"
        } catch {
            notes[key] = "bookmark failed: \(error.localizedDescription)"
        }
    }

    func clear(_ key: String) {
        urls[key] = nil
        notes[key] = nil
        defaults.removeObject(forKey: "bookmark." + key)
    }

    func path(_ key: String) -> String {
        return urls[key]?.path ?? ""
    }

    private func restore(_ key: String) {
        guard let data = defaults.data(forKey: "bookmark." + key) else { return }
        var stale = false
        if let url = try? URL(resolvingBookmarkData: data, options: [], relativeTo: nil, bookmarkDataIsStale: &stale) {
            _ = url.startAccessingSecurityScopedResource()
            urls[key] = url
        }
    }
}
