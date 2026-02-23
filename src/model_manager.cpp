#include "ave/model_manager.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef AVE_HAVE_CURL
#  include <curl/curl.h>
#endif

namespace ave {

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────
namespace {

std::string defaultModelsDir() {
    const char* home = std::getenv("HOME");
    if (home != nullptr) {
        return std::string(home) + "/.local/share/ave/models";
    }
    return "/tmp/ave_models";
}

bool ensureDir(const std::filesystem::path& p) {
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    return !ec;
}

bool fileExists(const std::filesystem::path& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

bool commandInPath(const std::string& cmd) {
    const char* pathEnv = std::getenv("PATH");
    if (pathEnv == nullptr) { return false; }
    std::string path(pathEnv);
    std::size_t start = 0;
    while (start <= path.size()) {
        std::size_t end = path.find(':', start);
        if (end == std::string::npos) { end = path.size(); }
        const std::string dir = path.substr(start, end - start);
        if (!dir.empty()) {
            if (fileExists(std::filesystem::path(dir) / cmd)) { return true; }
        }
        if (end == path.size()) { break; }
        start = end + 1;
    }
    return false;
}

std::string statePrefix(ModelState state) {
    switch (state) {
        case ModelState::NotDownloaded: return "[Not Downloaded]";
        case ModelState::Downloading:   return "[Downloading…]";
        case ModelState::Downloaded:    return "[Downloaded]";
        case ModelState::Converting:    return "[Converting…]";
        case ModelState::Converted:     return "[Compiled]";
        case ModelState::Optimizing:    return "[Optimizing…]";
        case ModelState::Optimized:     return "[Optimized]";
        case ModelState::Error:         return "[Error]";
    }
    return "";
}

// ─── CURL download helper ─────────────────────────────────────────
#ifdef AVE_HAVE_CURL

struct CurlWriteCtx {
    std::ofstream*           file  = nullptr;
    std::atomic<bool>*       cancel = nullptr;
};

static std::size_t curlWrite(void* ptr, std::size_t size, std::size_t nmemb, void* userData) {
    auto* ctx = static_cast<CurlWriteCtx*>(userData);
    if (ctx->cancel && ctx->cancel->load()) { return 0; }
    const std::size_t bytes = size * nmemb;
    ctx->file->write(static_cast<const char*>(ptr), static_cast<std::streamsize>(bytes));
    return bytes;
}

struct CurlProgressCtx {
    ModelProgressCb          cb;
    std::string              modelId;
    std::atomic<bool>*       cancel = nullptr;
};

static int curlProgress(void* clientP, curl_off_t dltotal, curl_off_t dlnow,
                        curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    auto* ctx = static_cast<CurlProgressCtx*>(clientP);
    if (ctx->cancel && ctx->cancel->load()) { return 1; }
    if (dltotal > 0 && ctx->cb) {
        const float progress = static_cast<float>(dlnow) / static_cast<float>(dltotal);
        ctx->cb(ctx->modelId, progress, "Downloading…");
    }
    return 0;
}

bool curlDownload(const std::string& url, const std::filesystem::path& destPath,
                  const ModelProgressCb& progressCb, const std::string& modelId,
                  std::atomic<bool>& cancelFlag, std::string& error) {
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        error = "Failed to initialise libcurl.";
        return false;
    }

    std::ofstream outFile(destPath, std::ios::binary | std::ios::trunc);
    if (!outFile.is_open()) {
        curl_easy_cleanup(curl);
        error = "Cannot open destination file for writing: " + destPath.string();
        return false;
    }

    CurlWriteCtx    writeCtx{&outFile, &cancelFlag};
    CurlProgressCtx progressCtx{progressCb, modelId, &cancelFlag};

    curl_easy_setopt(curl, CURLOPT_URL,              url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION,   1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS,       0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curlProgress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA,     &progressCtx);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,    curlWrite);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,        &writeCtx);
    curl_easy_setopt(curl, CURLOPT_USERAGENT,        "AMD Video Enhancer/1.0");

    const CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    outFile.close();

    if (cancelFlag.load()) {
        std::error_code ec;
        std::filesystem::remove(destPath, ec);
        error = "Download cancelled.";
        return false;
    }

    if (res != CURLE_OK) {
        std::error_code ec;
        std::filesystem::remove(destPath, ec);
        error = std::string("libcurl error: ") + curl_easy_strerror(res);
        return false;
    }

    return true;
}

#else  // AVE_HAVE_CURL

bool curlDownload(const std::string& url, const std::filesystem::path& destPath,
                  const ModelProgressCb& progressCb, const std::string& modelId,
                  std::atomic<bool>& cancelFlag, std::string& error) {
    (void)progressCb; (void)modelId; (void)cancelFlag; (void)destPath;
    error = "libcurl is not compiled in.  Cannot download " + url +
            "\n\nPlease place the model file manually at:\n  " + destPath.string();
    return false;
}

#endif // AVE_HAVE_CURL

// ─── Zip archive extraction ──────────────────────────────────────
// Requires `unzip` on PATH (standard on all Linux distributions).
bool extractFromZip(const std::filesystem::path& zipPath,
                    const std::string& internalPath,
                    const std::filesystem::path& destFile,
                    std::string& error) {
    if (!commandInPath("unzip")) {
        error = "'unzip' not found in PATH – cannot extract from archive.";
        return false;
    }

    // Extract with -j (junk paths) into a temp directory next to the zip.
    const std::filesystem::path tempDir = zipPath.parent_path() /
        (zipPath.stem().string() + "_extract_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    std::filesystem::create_directories(tempDir, ec);

    std::ostringstream cmd;
    cmd << "unzip -o -j "
        << "\"" << zipPath.string() << "\" "
        << "\"" << internalPath     << "\" "
        << "-d \"" << tempDir.string() << "\"";

    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::filesystem::remove_all(tempDir, ec);
        error = "unzip failed (exit " + std::to_string(rc) +
                ") extracting '" + internalPath + "' from archive.";
        return false;
    }

    const auto extracted = tempDir /
        std::filesystem::path(internalPath).filename();
    if (!fileExists(extracted)) {
        std::filesystem::remove_all(tempDir, ec);
        error = "Extracted file not found after unzip: " + extracted.string();
        return false;
    }

    std::filesystem::rename(extracted, destFile, ec);
    if (ec) {
        // Fallback: copy then remove
        std::filesystem::copy_file(extracted, destFile,
            std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(extracted, ec);
    }
    std::filesystem::remove_all(tempDir, ec);
    return true;
}

// ─── Quantized-op pre-flight scanner ────────────────────────────
// MiGraphX cannot lower quantized operators (QLinearConv, etc.) and will
// call abort() — exit 134 (SIGABRT) — instead of returning a clean non-zero
// code.  We scan the ONNX binary for known quantized op-type strings BEFORE
// invoking migraphx-driver so we can surface a clear ModelIncompatible error.
//
// Detection is via raw byte search (no protobuf library required).  The
// op_type values are stored as plain ASCII bytes inside the protobuf encoding,
// so a substring scan is reliable for these distinctive long strings.
// Reads only the first kScanBytes of the file — op defs appear well within the
// first few MiB for all typical SR / interpolation / denoising models.
static const std::array<const char*, 9> kQuantizedOpMarkers = {{
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "ConvInteger",
    "MatMulInteger",
    "QGemm",
}};

std::vector<std::string> scanOnnxQuantizedOps(
        const std::filesystem::path& onnxPath) {
    constexpr std::size_t kScanBytes = 4ULL * 1024 * 1024;  // 4 MiB
    std::vector<std::string> found;

    std::ifstream f(onnxPath, std::ios::binary);
    if (!f.is_open()) { return found; }

    std::vector<char> buf(kScanBytes);
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const auto bytesRead = static_cast<std::size_t>(f.gcount());

    for (const char* op : kQuantizedOpMarkers) {
        std::string opStr(op);
        const auto it = std::search(
            buf.data(), buf.data() + bytesRead,
            opStr.data(), opStr.data() + opStr.size());
        if (it != buf.data() + bytesRead) {
            found.emplace_back(op);
        }
    }
    return found;
}

// ─── MiGraphX compilation ─────────────────────────────────────────
// Uses the migraphx-driver command-line tool if present.
// The driver is shipped with ROCm and provides:
//   migraphx-driver compile --onnx <in.onnx> --output <out.mxr>
//   migraphx-driver compile --onnx <in.onnx> --gpu --output <out.mxr>
//
// PROGRESS NOTE: migraphx-driver emits no incremental progress to stdout/stderr,
// so we run the compile in a dedicated thread and send a heartbeat tick every
// 4 seconds, crawling progress from 5 % → 90 %.  The final 100 % is sent only
// after the process exits successfully.
bool migraphxCompile(const std::filesystem::path& onnxPath,
                     const std::filesystem::path& mxrPath,
                     bool gpuTune,
                     ModelPrecision precision,
                     const ModelProgressCb& progressCb,
                     const std::string& modelId,
                     std::string& error) {
    if (!commandInPath("migraphx-driver")) {
        error = "migraphx-driver not found in PATH.  ROCm must be installed.";
        return false;
    }

    // ── Pre-flight: quantized-op scan ────────────────────────────
    // Run before spawning migraphx-driver to avoid the SIGABRT (exit 134)
    // that the driver emits when it encounters quantized ops it cannot lower.
    {
        const auto quantOps = scanOnnxQuantizedOps(onnxPath);
        if (!quantOps.empty()) {
            std::ostringstream msg;
            msg << "Model contains quantized operators not supported by MiGraphX: ";
            for (std::size_t i = 0; i < quantOps.size(); ++i) {
                if (i > 0) { msg << ", "; }
                msg << quantOps[i];
            }
            msg << ".\n"
                << "Action: re-export the ONNX model in fp32 or fp16 precision "
                << "(remove QLinear/quantize ops) before compiling with MiGraphX.";
            error = msg.str();
            if (progressCb) {
                progressCb(modelId, 0.0f,
                    "Unsupported: quantized model (" + quantOps[0] + " …)");
            }
            return false;
        }
    }

    std::ostringstream cmd;
    cmd << "migraphx-driver compile"
        << " --onnx " << onnxPath.string()
        << " --output " << mxrPath.string();
    if (gpuTune)                          { cmd << " --gpu"; }
    if (precision == ModelPrecision::Fp16) { cmd << " --fp16"; }
    if (precision == ModelPrecision::Int8) { cmd << " --int8"; }

    const std::string cmdStr = cmd.str();

    // Shared state between compile thread and heartbeat ticker.
    std::atomic<bool>  compileDone{false};
    std::atomic<int>   compileExit{-1};

    if (progressCb) { progressCb(modelId, 0.05f, "Compiling with MiGraphX…"); }

    // Spawn compile work in a thread so this function can tick progress.
    std::thread compileThread([&]() {
        compileExit.store(std::system(cmdStr.c_str()));
        compileDone.store(true);
    });

    // Heartbeat: crawl from 5 % to 90 % while the driver runs.
    // Each tick is ~4 seconds; 3 % per tick reaches 90 % after ~28 ticks (~112 s).
    // Large models can take much longer, so we clamp at 90 % and wait.
    {
        constexpr auto kTickInterval = std::chrono::seconds(4);
        constexpr float kTickStep    = 0.03f;  // +3 % per tick
        constexpr float kMaxProgress = 0.90f;
        float progress = 0.05f;

        while (!compileDone.load()) {
            std::this_thread::sleep_for(kTickInterval);
            if (compileDone.load()) { break; }
            progress = std::min(progress + kTickStep, kMaxProgress);
            if (progressCb) {
                progressCb(modelId, progress, "Compiling with MiGraphX… (this can take several minutes)");
            }
        }
    }

    compileThread.join();

    const int rc = compileExit.load();
    if (rc != 0) {
        std::ostringstream msg;
        // WIFEXITED / WEXITSTATUS semantics: std::system() returns the raw
        // wait-status on POSIX.  Signals appear as 128 + signum.
        if (rc > 128) {
            const int sig = rc - 128;
            msg << "migraphx-driver killed by signal " << sig;
            if (sig == 6) {
                msg << " (SIGABRT).  This usually means MiGraphX encountered "
                    << "an operator it cannot lower.  "
                    << "Common causes: quantized operators (QLinearConv, "
                    << "QuantizeLinear, …), opset > " << 19
                    << ", or an internal MiGraphX assertion failure.\n"
                    << "The quantized-op pre-flight scan did not catch this model — "
                    << "please report the model name so the scanner can be updated.";
            } else if (sig == 11) {
                msg << " (SIGSEGV – internal MiGraphX crash).";
            } else {
                msg << ".";
            }
        } else {
            msg << "migraphx-driver exited with code " << rc << ".";
        }
        error = msg.str();
        if (progressCb) {
            progressCb(modelId, 0.0f,
                "Compilation failed (exit " + std::to_string(rc) + ")");
        }
        return false;
    }

    if (!fileExists(mxrPath)) {
        error = "migraphx-driver ran but output file not found: " + mxrPath.string();
        if (progressCb) { progressCb(modelId, 0.0f, "Compilation failed: output .mxr missing"); }
        return false;
    }

    if (progressCb) { progressCb(modelId, 1.0f, "Compilation complete."); }
    return true;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────
// std::string toString(ModelState)
// ─────────────────────────────────────────────────────────────────
std::string toString(ModelState state) {
    switch (state) {
        case ModelState::NotDownloaded: return "not_downloaded";
        case ModelState::Downloading:   return "downloading";
        case ModelState::Downloaded:    return "downloaded";
        case ModelState::Converting:    return "converting";
        case ModelState::Converted:     return "converted";
        case ModelState::Optimizing:    return "optimizing";
        case ModelState::Optimized:     return "optimized";
        case ModelState::Error:         return "error";
    }
    return "unknown";
}

// ─────────────────────────────────────────────────────────────────
// ModelManager::Impl
// ─────────────────────────────────────────────────────────────────
struct ModelManager::Impl {
    mutable std::mutex                             mtx;
    std::string                                    modelsDir;
    std::unordered_map<std::string, ManagedModel>  records;   // keyed by modelId
    std::unordered_map<std::string, std::shared_ptr<std::atomic<bool>>> cancelFlags;

    ManagedModel& getOrCreate(const std::string& modelId) {
        auto it = records.find(modelId);
        if (it == records.end()) {
            const ModelEntry* entry = catalogEntryById(modelId);
            assert(entry != nullptr);
            ManagedModel m;
            m.entry = *entry;
            records.emplace(modelId, std::move(m));
        }
        return records.at(modelId);
    }

    std::filesystem::path downloadedDir()  const { return std::filesystem::path(modelsDir) / "downloaded"; }
    std::filesystem::path convertedDir()   const { return std::filesystem::path(modelsDir) / "migraphx"; }
    std::filesystem::path optimizedDir()   const { return std::filesystem::path(modelsDir) / "optimized"; }

    void scanAndUpdate(ManagedModel& m) {
        const std::string& id = m.entry.id;

        // Primary downloaded file
        if (!m.entry.filename.empty()) {
            const auto p = downloadedDir() / m.entry.filename;
            m.downloadedPath = fileExists(p) ? p.string() : "";
        }

        // Auxiliary file (NCNN .bin)
        if (!m.entry.filenameAux.empty()) {
            const auto p = downloadedDir() / m.entry.filenameAux;
            m.downloadedPathAux = fileExists(p) ? p.string() : "";
        }

        // Compiled .mxr (regular)
        {
            const auto p = convertedDir() / (id + ".mxr");
            m.convertedPath = fileExists(p) ? p.string() : "";
        }

        // GPU-tuned .mxr
        {
            const auto p = optimizedDir() / (id + "_tuned.mxr");
            m.optimizedPath = fileExists(p) ? p.string() : "";
        }

        // Derive state (do not overwrite transient states like Downloading)
        if (m.state == ModelState::Downloading ||
            m.state == ModelState::Converting  ||
            m.state == ModelState::Optimizing) {
            return;
        }

        if (!m.optimizedPath.empty()) {
            m.state = ModelState::Optimized;
        } else if (!m.convertedPath.empty()) {
            m.state = ModelState::Converted;
        } else if (!m.downloadedPath.empty()) {
            m.state = ModelState::Downloaded;
        } else if (m.entry.filename.empty() && m.entry.downloadUrl.empty()) {
            // Built-in / parametric model – always "Downloaded"
            m.state = ModelState::Downloaded;
            m.downloadedPath = "(builtin)";
        } else {
            m.state = ModelState::NotDownloaded;
        }
    }
};

// ─────────────────────────────────────────────────────────────────
// ModelManager public interface
// ─────────────────────────────────────────────────────────────────
ModelManager::ModelManager() : impl_(std::make_unique<Impl>()) {
    impl_->modelsDir = defaultModelsDir();
    ensureDir(impl_->downloadedDir());
    ensureDir(impl_->convertedDir());
    ensureDir(impl_->optimizedDir());

    for (const auto& entry : builtinModelCatalog()) {
        ManagedModel m;
        m.entry = entry;
        impl_->records.emplace(entry.id, std::move(m));
    }

    refresh();
}

ModelManager::~ModelManager() = default;

void ModelManager::setModelsDirectory(const std::string& dir) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->modelsDir = dir;
    ensureDir(impl_->downloadedDir());
    ensureDir(impl_->convertedDir());
    ensureDir(impl_->optimizedDir());
}

std::string ModelManager::modelsDirectory() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    return impl_->modelsDir;
}

void ModelManager::refresh() {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    for (auto& [id, record] : impl_->records) {
        impl_->scanAndUpdate(record);
    }
}

std::vector<ManagedModel> ModelManager::allModels() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    std::vector<ManagedModel> out;
    out.reserve(impl_->records.size());
    for (const auto& [id, m] : impl_->records) {
        out.push_back(m);
    }
    // Stable order: preserve catalog order
    std::sort(out.begin(), out.end(), [](const ManagedModel& a, const ManagedModel& b) {
        const auto& cat = builtinModelCatalog();
        auto pos = [&](const std::string& id_) {
            for (std::size_t i = 0; i < cat.size(); ++i) {
                if (cat[i].id == id_) return static_cast<int>(i);
            }
            return static_cast<int>(cat.size());
        };
        return pos(a.entry.id) < pos(b.entry.id);
    });
    return out;
}

std::vector<ManagedModel> ModelManager::modelsForStage(StageKind kind) const {
    auto all = allModels();
    all.erase(std::remove_if(all.begin(), all.end(),
                             [kind](const ManagedModel& m) { return m.entry.stage != kind; }),
              all.end());
    return all;
}

std::optional<ManagedModel> ModelManager::findModel(const std::string& modelId) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->records.find(modelId);
    if (it == impl_->records.end()) { return std::nullopt; }
    return it->second;
}

std::optional<std::string> ModelManager::bestPathForModel(const std::string& modelId) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->records.find(modelId);
    if (it == impl_->records.end()) { return std::nullopt; }
    const auto& m = it->second;
    if (!m.optimizedPath.empty())  return m.optimizedPath;
    if (!m.convertedPath.empty())  return m.convertedPath;
    if (!m.downloadedPath.empty()) return m.downloadedPath;
    return std::nullopt;
}

bool ModelManager::startDownload(const std::string& modelId,
                                  const ModelProgressCb& progressCb,
                                  const ModelStateCb&    stateCb,
                                  std::string&           error) {
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto it = impl_->records.find(modelId);
        if (it == impl_->records.end()) {
            error = "Unknown model id: " + modelId;
            return false;
        }
        const auto& m = it->second;
        if (m.state == ModelState::Downloading) {
            error = "Already downloading.";
            return false;
        }
        if (m.state == ModelState::Downloaded ||
            m.state == ModelState::Converted  ||
            m.state == ModelState::Optimized) {
            error = "Model already available at: " + m.downloadedPath;
            return false;
        }
        if (m.entry.downloadUrl.empty()) {
            error = "Model has no download URL (built-in / parametric).";
            return false;
        }

        // Set transient state
        impl_->records[modelId].state = ModelState::Downloading;
        impl_->cancelFlags[modelId] = std::make_shared<std::atomic<bool>>(false);
    }

    if (stateCb) { stateCb(modelId, ModelState::Downloading); }

    // Capture copies for the thread
    const ModelEntry entry  = *catalogEntryById(modelId);
    const std::filesystem::path dlDir = impl_->downloadedDir();

    // Capture cancelFlag as shared_ptr so the thread holds a stable reference
    // even if the cancelFlags map rehashes.
    std::shared_ptr<std::atomic<bool>> cancelFlag;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        cancelFlag = impl_->cancelFlags.at(modelId);
    }

    std::thread([this, entry, dlDir, progressCb, stateCb, modelId, cancelFlag]() mutable {
        std::string err;

        const bool isZipArchive = !entry.archiveSubPath.empty();

        // For zip archives, download to a temporary file; for direct files,
        // download straight to the final destination.
        const auto destPath = isZipArchive
            ? dlDir / (entry.id + "_archive.zip")
            : dlDir / entry.filename;

        bool ok = curlDownload(entry.downloadUrl, destPath,
                               progressCb, modelId, *cancelFlag, err);

        if (ok && isZipArchive) {
            if (progressCb) { progressCb(entry.id, 0.95f, "Extracting from archive…"); }
            ok = extractFromZip(destPath, entry.archiveSubPath,
                                dlDir / entry.filename, err);
            if (ok && !entry.archiveSubPathAux.empty() && !entry.filenameAux.empty()) {
                ok = extractFromZip(destPath, entry.archiveSubPathAux,
                                    dlDir / entry.filenameAux, err);
            }
            // Remove the temporary archive regardless of extraction outcome.
            std::error_code removeEc;
            std::filesystem::remove(destPath, removeEc);
        } else if (ok && !entry.downloadUrlAux.empty()) {
            // Direct download of a second file (NCNN .bin without zip)
            const auto destAux = dlDir / entry.filenameAux;
            ok = curlDownload(entry.downloadUrlAux, destAux,
                              progressCb, modelId, *cancelFlag, err);
        }

        ModelState finalState = ModelState::Error;
        {
            std::lock_guard<std::mutex> lock(impl_->mtx);
            auto& m = impl_->records[modelId];
            if (ok) {
                impl_->scanAndUpdate(m);
            } else {
                m.state        = ModelState::Error;
                m.errorMessage = err;
            }
            finalState = m.state;
        }

        if (stateCb) { stateCb(modelId, finalState); }
        if (progressCb) {
            progressCb(modelId, 1.0f, ok ? "Download complete." : ("Download failed: " + err));
        }
    }).detach();

    return true;
}

void ModelManager::cancelDownload(const std::string& modelId) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->cancelFlags.find(modelId);
    if (it != impl_->cancelFlags.end() && it->second) {
        it->second->store(true);
    }
}

bool ModelManager::convertToMiGraphX(const std::string& modelId,
                                      const ModelProgressCb& progressCb,
                                      const ModelStateCb&    stateCb,
                                      std::string&           error,
                                      std::optional<ModelPrecision> precisionOverride) {
    std::string onnxPath;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto it = impl_->records.find(modelId);
        if (it == impl_->records.end()) {
            error = "Unknown model id: " + modelId;
            return false;
        }
        const auto& m = it->second;
        if (m.downloadedPath.empty() || m.downloadedPath == "(builtin)") {
            error = "Model not yet downloaded – cannot compile.";
            return false;
        }
        if (m.entry.sourceFormat != ModelFormat::Onnx) {
            error = "Only ONNX models can be compiled with MiGraphX.";
            return false;
        }
        onnxPath = m.downloadedPath;
        impl_->records[modelId].state = ModelState::Converting;
    }
    if (stateCb) { stateCb(modelId, ModelState::Converting); }

    // Precision: caller override → entry catalog value.
    // (Global setting applied by caller before passing precisionOverride.)
    ModelPrecision precision = ModelPrecision::Fp32;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        const auto it2 = impl_->records.find(modelId);
        if (it2 != impl_->records.end()) {
            precision = precisionOverride.value_or(it2->second.entry.precision);
        }
    }

    const auto mxrPath = impl_->convertedDir() / (modelId + ".mxr");
    const bool ok = migraphxCompile(onnxPath, mxrPath, /*gpuTune=*/false,
                                    precision, progressCb, modelId, error);

    ModelState finalState = ModelState::Error;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto& m = impl_->records[modelId];
        if (ok) {
            impl_->scanAndUpdate(m);
        } else {
            m.state        = ModelState::Error;
            m.errorMessage = error;
        }
        finalState = m.state;
    }
    if (stateCb) { stateCb(modelId, finalState); }
    return ok;
}

bool ModelManager::optimizeForHardware(const std::string& modelId,
                                        const ModelProgressCb& progressCb,
                                        const ModelStateCb&    stateCb,
                                        std::string&           error,
                                        std::optional<ModelPrecision> precisionOverride) {
    std::string onnxPath;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto it = impl_->records.find(modelId);
        if (it == impl_->records.end()) {
            error = "Unknown model id: " + modelId;
            return false;
        }
        const auto& m = it->second;
        if (m.downloadedPath.empty() || m.downloadedPath == "(builtin)") {
            error = "Model not yet downloaded – cannot optimise.";
            return false;
        }
        if (m.entry.sourceFormat != ModelFormat::Onnx) {
            error = "Only ONNX models can be optimised with MiGraphX.";
            return false;
        }
        onnxPath = m.downloadedPath;
        impl_->records[modelId].state = ModelState::Optimizing;
    }
    if (stateCb) { stateCb(modelId, ModelState::Optimizing); }

    ModelPrecision precision = ModelPrecision::Fp32;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        const auto it2 = impl_->records.find(modelId);
        if (it2 != impl_->records.end()) {
            precision = precisionOverride.value_or(it2->second.entry.precision);
        }
    }

    const auto mxrPath = impl_->optimizedDir() / (modelId + "_tuned.mxr");
    const bool ok = migraphxCompile(onnxPath, mxrPath, /*gpuTune=*/true,
                                    precision, progressCb, modelId, error);

    ModelState finalState = ModelState::Error;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto& m = impl_->records[modelId];
        if (ok) {
            impl_->scanAndUpdate(m);
        } else {
            m.state        = ModelState::Error;
            m.errorMessage = error;
        }
        finalState = m.state;
    }
    if (stateCb) { stateCb(modelId, finalState); }
    return ok;
}

std::string ModelManager::modelDropdownLabel(const std::string& modelId) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->records.find(modelId);
    if (it == impl_->records.end()) { return modelId; }
    const auto& m = it->second;
    return statePrefix(m.state) + " " + m.entry.displayName;
}

std::vector<ModelManager::DropdownEntry>
ModelManager::dropdownEntriesForStage(StageKind kind) const {
    const auto models = modelsForStage(kind);
    std::vector<DropdownEntry> out;
    out.reserve(models.size());

    // Sort: inference-ready first (optimized > converted > downloaded), then by name
    for (const auto& m : models) {
        DropdownEntry de;
        de.modelId       = m.entry.id;
        de.label         = statePrefix(m.state) + " " + m.entry.displayName;
        de.inferenceReady = (m.state == ModelState::Downloaded  ||
                             m.state == ModelState::Converted   ||
                             m.state == ModelState::Optimized);
        out.push_back(de);
    }

    std::stable_sort(out.begin(), out.end(), [](const DropdownEntry& a, const DropdownEntry& b) {
        if (a.inferenceReady != b.inferenceReady) { return a.inferenceReady > b.inferenceReady; }
        return false;
    });

    return out;
}

}  // namespace ave
