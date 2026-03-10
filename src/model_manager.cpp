#include "ave/model_manager.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#  include <sys/wait.h>
#endif

#ifdef AVE_HAVE_CURL
#  include <curl/curl.h>
#endif

#include "ave/frame_io.hpp"

#ifdef AVE_HAVE_MIGRAPHX
#  include <migraphx/migraphx.hpp>
#endif

namespace ave {

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────
namespace {

// Small fixed-shape fallback used only for explicit "convert model" actions
// where no real video dimensions are available yet. Runtime auto-compile still
// waits for the actual input frame size before producing inference artifacts.
constexpr int kDefaultBaselineCompileWidth = 192;
constexpr int kDefaultBaselineCompileHeight = 192;

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

bool hasMxrExtension(const std::string& path) {
    if (path.size() < 4) { return false; }
    const std::string ext = path.substr(path.size() - 4);
    return ext == ".mxr" || ext == ".MXR";
}

std::string shellQuote(const std::string& value) {
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('\'');
    return out;
}

std::string trimLine(std::string line) {
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    return line;
}

std::string statePrefix(ModelState state) {
    switch (state) {
        case ModelState::NotDownloaded: return "[Not Downloaded]";
        case ModelState::Downloading:   return "[Downloading…]";
        case ModelState::Downloaded:    return "[Downloaded]";
        case ModelState::Converting:    return "[Converting…]";
        case ModelState::Converted:     return "[Compiled]";
        case ModelState::Error:         return "[Error]";
    }
    return "";
}

bool isSupportedMiGraphXCompilePrecision(ModelPrecision precision) {
    return precision == ModelPrecision::Fp32 ||
           precision == ModelPrecision::Fp16 ||
           precision == ModelPrecision::Int8;
}

bool isMiGraphXCompileTimeout(const std::string& error) {
    return error.find("migraphx-driver timed out after") != std::string::npos;
}

bool shouldRetryFp32AfterTimeout(ModelPrecision requestedPrecision,
                                 const std::string& error) {
    return requestedPrecision == ModelPrecision::Fp16 &&
           isMiGraphXCompileTimeout(error);
}

std::string compilePrecisionTag(ModelPrecision precision) {
    switch (precision) {
        case ModelPrecision::Fp32: return "fp32";
        case ModelPrecision::Fp16: return "fp16";
        case ModelPrecision::Int8: return "int8";
    }
    return "unknown";
}

std::string compiledArtifactStem(const std::string& modelId,
                                 ModelPrecision precision,
                                 std::optional<int> width = std::nullopt,
                                 std::optional<int> height = std::nullopt) {
    std::ostringstream stem;
    stem << modelId;
    if (width.has_value() && height.has_value()) {
        stem << "_" << *width << "x" << *height;
    }
    if (precision != ModelPrecision::Fp32) {
        stem << "_" << compilePrecisionTag(precision);
    }
    return stem.str();
}

std::filesystem::path compiledArtifactPath(const std::filesystem::path& dir,
                                           const std::string& modelId,
                                           ModelPrecision precision,
                                           std::optional<int> width = std::nullopt,
                                           std::optional<int> height = std::nullopt) {
    return dir / (compiledArtifactStem(modelId, precision, width, height) + ".mxr");
}

int readPositiveIntEnv(const char* name, int defaultValue, int minimumValue, int maximumValue) {
    int value = defaultValue;
    if (const char* raw = std::getenv(name); raw != nullptr) {
        try {
            value = std::stoi(raw);
        } catch (...) {
            value = defaultValue;
        }
    }
    value = std::max(value, minimumValue);
    value = std::min(value, maximumValue);
    return value;
}

int baselineCompileWidth() {
    return readPositiveIntEnv("AVE_MIGRAPHX_BASELINE_WIDTH",
                              kDefaultBaselineCompileWidth, 32, 4096);
}

int baselineCompileHeight() {
    return readPositiveIntEnv("AVE_MIGRAPHX_BASELINE_HEIGHT",
                              kDefaultBaselineCompileHeight, 32, 4096);
}

#ifdef AVE_HAVE_MIGRAPHX

constexpr int kDefaultInt8CalibrationFrames = 8;

struct CalibrationSample {
    std::vector<migraphx::argument> args;
    migraphx::program_parameters params;
};

bool extractCalibrationFramesRgb24(const std::string& videoPath,
                                   int width,
                                   int height,
                                   int requestedFrames,
                                   std::vector<std::vector<std::uint8_t>>& frames,
                                   std::string& error) {
    if (!commandInPath("ffmpeg")) {
        error = "ffmpeg is required to extract calibration frames for int8 compilation.";
        return false;
    }
    if (requestedFrames <= 0) {
        error = "Requested calibration frame count must be positive.";
        return false;
    }

    const std::size_t frameBytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u;
    const std::string scaleArg =
        "scale=" + std::to_string(width) + ":" + std::to_string(height) + ":flags=bicubic";
    const std::string cmd =
        "ffmpeg -hide_banner -loglevel error -i " + shellQuote(videoPath) +
        " -vf " + shellQuote(scaleArg) +
        " -frames:v " + std::to_string(requestedFrames) +
        " -f rawvideo -pix_fmt rgb24 pipe:1 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        error = "Failed to start ffmpeg while extracting int8 calibration frames.";
        return false;
    }

    frames.clear();
    std::vector<std::uint8_t> buffer(frameBytes);
    while (static_cast<int>(frames.size()) < requestedFrames) {
        std::size_t totalRead = 0u;
        while (totalRead < frameBytes) {
            const std::size_t n = std::fread(buffer.data() + totalRead, 1u,
                                             frameBytes - totalRead, pipe);
            if (n == 0u) {
                break;
            }
            totalRead += n;
        }
        if (totalRead != frameBytes) {
            break;
        }
        frames.push_back(buffer);
    }

    const int rc = pclose(pipe);
    if (rc != 0 && frames.empty()) {
        error = "ffmpeg failed while extracting int8 calibration frames from: " + videoPath;
        return false;
    }
    if (frames.empty()) {
        error = "No calibration frames could be extracted from: " + videoPath;
        return false;
    }
    return true;
}

bool shapeLooksLikeImageTensor(const migraphx::shape& shape, int width, int height) {
    const auto dims = shape.lengths();
    if (dims.size() == 4u) {
        if (dims[0] != 1u) {
            return false;
        }
        if (dims[1] <= 4u) {
            return (dims[1] == 3u || dims[1] == 1u) &&
                   dims[2] == static_cast<std::size_t>(height) &&
                   dims[3] == static_cast<std::size_t>(width);
        }
        if (dims[3] <= 4u) {
            return (dims[3] == 3u || dims[3] == 1u) &&
                   dims[1] == static_cast<std::size_t>(height) &&
                   dims[2] == static_cast<std::size_t>(width);
        }
        return false;
    }
    if (dims.size() == 3u) {
        if (dims[0] <= 4u) {
            return (dims[0] == 3u || dims[0] == 1u) &&
                   dims[1] == static_cast<std::size_t>(height) &&
                   dims[2] == static_cast<std::size_t>(width);
        }
        if (dims[2] <= 4u) {
            return (dims[2] == 3u || dims[2] == 1u) &&
                   dims[0] == static_cast<std::size_t>(height) &&
                   dims[1] == static_cast<std::size_t>(width);
        }
    }
    return false;
}

bool fillCalibrationArgument(const std::vector<std::uint8_t>& rgbFrame,
                             int width,
                             int height,
                             const migraphx::shape& shape,
                             migraphx::argument& arg,
                             std::string& error) {
    if (shape.type() == migraphx_shape_float_type) {
        std::vector<float> tensor;
        frame_io::rgb24ToNchwFp32(rgbFrame.data(), width, height, tensor);
        const std::size_t bytes = tensor.size() * sizeof(float);
        if (bytes != shape.bytes()) {
            error = "Calibration tensor byte size mismatch for fp32 input.";
            return false;
        }
        std::memcpy(arg.data(), tensor.data(), bytes);
        return true;
    }

    if (shape.type() == migraphx_shape_half_type) {
        std::vector<std::uint16_t> tensor;
        frame_io::rgb24ToNchwFp16(rgbFrame.data(), width, height, tensor);
        const std::size_t bytes = tensor.size() * sizeof(std::uint16_t);
        if (bytes != shape.bytes()) {
            error = "Calibration tensor byte size mismatch for fp16 input.";
            return false;
        }
        std::memcpy(arg.data(), tensor.data(), bytes);
        return true;
    }

    error = "Unsupported MiGraphX input dtype for int8 calibration: "
          + std::to_string(static_cast<int>(shape.type()));
    return false;
}

bool buildInt8CalibrationData(const migraphx::program& prog,
                              const std::string& calibrationVideoPath,
                              int width,
                              int height,
                              std::vector<CalibrationSample>& samples,
                              std::string& error) {
    const int requestedFrames = readPositiveIntEnv(
        "AVE_MIGRAPHX_INT8_CALIBRATION_FRAMES", kDefaultInt8CalibrationFrames, 1, 64);

    std::vector<std::vector<std::uint8_t>> rgbFrames;
    if (!extractCalibrationFramesRgb24(calibrationVideoPath, width, height, requestedFrames,
                                       rgbFrames, error)) {
        return false;
    }

    const auto paramShapes = prog.get_parameter_shapes();
    std::string primaryInputName;
    for (const char* rawName : paramShapes.names()) {
        if (rawName == nullptr) {
            continue;
        }
        const std::string name(rawName);
        if (name.empty() || name.find("#output") != std::string::npos) {
            continue;
        }
        if (shapeLooksLikeImageTensor(paramShapes[name.c_str()], width, height)) {
            primaryInputName = name;
            break;
        }
    }

    if (primaryInputName.empty()) {
        for (const char* rawName : paramShapes.names()) {
            if (rawName == nullptr) {
                continue;
            }
            const std::string name(rawName);
            if (!name.empty() && name.find("#output") == std::string::npos) {
                primaryInputName = name;
                break;
            }
        }
    }

    if (primaryInputName.empty()) {
        error = "MiGraphX int8 calibration could not find a usable input tensor.";
        return false;
    }

    samples.clear();
    samples.reserve(rgbFrames.size());
    for (const auto& frame : rgbFrames) {
        CalibrationSample sample;
        sample.args.reserve(paramShapes.size());

        for (const char* rawName : paramShapes.names()) {
            if (rawName == nullptr) {
                continue;
            }
            const std::string name(rawName);
            if (name.empty() || name.find("#output") != std::string::npos) {
                continue;
            }

            const auto shape = paramShapes[name.c_str()];
            sample.args.emplace_back(shape);
            auto& arg = sample.args.back();

            if (name == primaryInputName) {
                if (!fillCalibrationArgument(frame, width, height, shape, arg, error)) {
                    return false;
                }
            } else {
                std::memset(arg.data(), 0, shape.bytes());
            }

            sample.params.add(name.c_str(), arg);
        }

        samples.push_back(std::move(sample));
    }

    return !samples.empty();
}

#endif  // AVE_HAVE_MIGRAPHX

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

    // ── Check HTTP status code before cleaning up ──────────────
    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

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

    // ── Validate HTTP response code ────────────────────────────
    if (httpCode != 200) {
        std::error_code ec;
        std::filesystem::remove(destPath, ec);
        error = "HTTP error " + std::to_string(httpCode) +
                " while downloading " + url;
        return false;
    }

    // ── Validate downloaded file size (models are always > 1 KB) ─
    {
        std::error_code ec;
        const auto fileSize = std::filesystem::file_size(destPath, ec);
        if (ec || fileSize < 1024) {
            std::filesystem::remove(destPath, ec);
            error = "Downloaded file is too small (" +
                    std::to_string(ec ? 0 : fileSize) +
                    " bytes); the URL may be invalid: " + url;
            return false;
        }
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

void appendDriverDimParamFallback(std::ostringstream& cmd,
                                  int compileWidth,
                                  int compileHeight) {
    cmd << " --batch 1"
        << " --dim-param @batch_size 1"
        << " --dim-param @b 1"
        << " --dim-param @n 1"
        << " --dim-param @width " << compileWidth
        << " --dim-param @w " << compileWidth
        << " --dim-param @x " << compileWidth
        << " --dim-param @cols " << compileWidth
        << " --dim-param @height " << compileHeight
        << " --dim-param @h " << compileHeight
        << " --dim-param @y " << compileHeight
        << " --dim-param @rows " << compileHeight;

    if (compileWidth == compileHeight) {
        const std::string fixedDynDim =
            "{min:" + std::to_string(compileWidth) +
            ", max:" + std::to_string(compileWidth) +
            ", optimals:[" + std::to_string(compileWidth) + "]}";
        cmd << " --default-dyn-dim " << shellQuote(fixedDynDim);
    }
}

#ifdef AVE_HAVE_MIGRAPHX
bool appendDriverInputDims(std::ostringstream& cmd,
                           const std::filesystem::path& onnxPath,
                           int compileWidth,
                           int compileHeight,
                           std::string& error) {
    migraphx::onnx_options probeOpts;
    probeOpts.set_default_dim_value(1);

    migraphx::program prog;
    try {
        prog = migraphx::parse_onnx(onnxPath.string().c_str(), probeOpts);
    } catch (const std::exception& ex) {
        error = std::string("MiGraphX could not inspect ONNX inputs for ")
              + onnxPath.string() + ": " + ex.what();
        return false;
    }

    const auto paramShapes = prog.get_parameter_shapes();
    const auto w = static_cast<std::size_t>(compileWidth);
    const auto h = static_cast<std::size_t>(compileHeight);
    bool appendedAny = false;

    for (const char* rawName : paramShapes.names()) {
        if (rawName == nullptr) { continue; }
        const std::string name(rawName);
        if (name.empty() || name.find("#output") != std::string::npos) { continue; }

        auto dims = paramShapes[name.c_str()].lengths();
        if (dims.empty()) { continue; }

        if (dims.size() == 4) {
            dims[0] = 1;
            if (dims[1] <= 4) {
                dims[2] = h;
                dims[3] = w;
            } else if (dims[3] <= 4) {
                dims[1] = h;
                dims[2] = w;
            } else {
                dims[2] = h;
                dims[3] = w;
            }
        } else if (dims.size() == 3) {
            if (dims[0] <= 4) {
                dims[1] = h;
                dims[2] = w;
            } else if (dims[2] <= 4) {
                dims[0] = h;
                dims[1] = w;
            } else {
                dims[1] = h;
                dims[2] = w;
            }
        } else {
            continue;
        }

        cmd << " --input-dim " << shellQuote("@" + name);
        for (const auto dim : dims) {
            cmd << ' ' << dim;
        }
        appendedAny = true;
    }

    if (!appendedAny) {
        error = "MiGraphX found no usable input tensors in " + onnxPath.string();
        return false;
    }

    return true;
}

bool compileWithMigraphxLibrary(const std::filesystem::path& onnxPath,
                                const std::filesystem::path& mxrPath,
                                int compileWidth,
                                int compileHeight,
                                ModelPrecision compilePrecision,
                                std::optional<std::string> calibrationVideoPath,
                                const ModelProgressCb& progressCb,
                                const std::string& modelId,
                                std::string& error) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
    try {
        if (compilePrecision == ModelPrecision::Fp16) {
            error = "MiGraphX C++ runtime fallback only supports fp32 artifacts; "
                    "install migraphx-driver to compile " + compilePrecisionTag(compilePrecision) + " artifacts.";
            if (progressCb) {
                progressCb(modelId, 0.0f,
                    "Compilation failed: " + compilePrecisionTag(compilePrecision)
                    + " requires migraphx-driver");
            }
            return false;
        }
        if (compilePrecision == ModelPrecision::Int8 &&
            (!calibrationVideoPath.has_value() || calibrationVideoPath->empty())) {
            error = "MiGraphX int8 compilation requires a calibration video path.";
            if (progressCb) {
                progressCb(modelId, 0.0f, "Compilation failed: int8 requires calibration video");
            }
            return false;
        }

        const std::string onnxStr = onnxPath.string();
        const std::string mxrStr = mxrPath.string();

        if (progressCb) { progressCb(modelId, 0.02f, "Reading ONNX model…"); }
        migraphx::onnx_options firstOpts;
        firstOpts.set_default_dim_value(1);
        auto prog = migraphx::parse_onnx(onnxStr.c_str(), firstOpts);

        if (progressCb) {
            progressCb(modelId, 0.05f,
                "Setting input dimensions ("
                + std::to_string(compileWidth) + "x"
                + std::to_string(compileHeight) + ")…");
        }

        migraphx::onnx_options finalOpts;
        finalOpts.set_default_dim_value(1);
        const auto paramShapes = prog.get_parameter_shapes();
        const auto w = static_cast<std::size_t>(compileWidth);
        const auto h = static_cast<std::size_t>(compileHeight);

        for (const char* rawName : paramShapes.names()) {
            if (rawName == nullptr) { continue; }
            const std::string nm(rawName);
            if (nm.empty() || nm.find("#output") != std::string::npos) { continue; }
            auto dims = paramShapes[nm.c_str()].lengths();
            if (dims.empty()) { continue; }

            if (dims.size() == 4) {
                dims[0] = 1;
                if (dims[1] <= 4) {
                    dims[2] = h;
                    dims[3] = w;
                } else if (dims[3] <= 4) {
                    dims[1] = h;
                    dims[2] = w;
                } else {
                    dims[2] = h;
                    dims[3] = w;
                }
                finalOpts.set_input_parameter_shape(nm.c_str(), dims);
            } else if (dims.size() == 3) {
                if (dims[0] <= 4) {
                    dims[1] = h;
                    dims[2] = w;
                } else if (dims[2] <= 4) {
                    dims[0] = h;
                    dims[1] = w;
                } else {
                    dims[1] = h;
                    dims[2] = w;
                }
                finalOpts.set_input_parameter_shape(nm.c_str(), dims);
            }
        }

        prog = migraphx::parse_onnx(onnxStr.c_str(), finalOpts);

        if (compilePrecision == ModelPrecision::Int8) {
            if (progressCb) {
                progressCb(modelId, 0.25f, "Extracting calibration frames for int8…");
            }
            std::vector<CalibrationSample> calibrationSamples;
            if (!buildInt8CalibrationData(prog, *calibrationVideoPath,
                                          compileWidth, compileHeight,
                                          calibrationSamples, error)) {
                if (progressCb) {
                    progressCb(modelId, 0.0f, "Compilation failed: int8 calibration setup");
                }
                return false;
            }

            if (progressCb) {
                progressCb(modelId, 0.40f,
                    "Quantizing to int8 with "
                    + std::to_string(calibrationSamples.size()) + " calibration frame(s)…");
            }

            migraphx::quantize_int8_options int8Options;
            int8Options.add_op_name("dot");
            int8Options.add_op_name("convolution");
            for (const auto& sample : calibrationSamples) {
                int8Options.add_calibration_data(sample.params);
            }
            migraphx::quantize_int8(prog, migraphx::target("gpu"), int8Options);
        }

        if (progressCb) {
            progressCb(modelId, -1.0f, "Compiling with MiGraphX C++ runtime…");
        }
        migraphx::compile_options copts;
        copts.set_offload_copy(true);
        prog.compile(migraphx::target("gpu"), copts);

        if (progressCb) { progressCb(modelId, 0.95f, "Saving compiled model…"); }
        migraphx::save(prog, mxrStr.c_str());

        if (!fileExists(mxrPath)) {
            error = "Compilation succeeded but output file not found: " + mxrStr;
            return false;
        }

        if (progressCb) { progressCb(modelId, 1.0f, "Compilation complete."); }
        return true;
    } catch (const std::exception& ex) {
        error = std::string("MiGraphX compilation failed: ") + ex.what();
        if (progressCb) { progressCb(modelId, 0.0f, "Compilation failed"); }
        return false;
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
}
#endif

bool compileWithMigraphxDriver(const std::filesystem::path& onnxPath,
                               const std::filesystem::path& mxrPath,
                               int compileWidth,
                               int compileHeight,
                               ModelPrecision compilePrecision,
                               const ModelProgressCb& progressCb,
                               const std::string& modelId,
                               std::string& error) {
    if (!commandInPath("migraphx-driver")) {
        error = "migraphx-driver not found in PATH. ROCm must be installed.";
        return false;
    }

    if (!isSupportedMiGraphXCompilePrecision(compilePrecision)) {
        error = "MiGraphX compile precision '" + compilePrecisionTag(compilePrecision)
              + "' is not supported by this app yet.";
        return false;
    }
    if (compilePrecision == ModelPrecision::Int8) {
        error = "MiGraphX int8 compilation uses the C++ runtime path because calibration data "
                "must be supplied explicitly.";
        return false;
    }

    if (progressCb) { progressCb(modelId, 0.02f, "Inspecting ONNX input tensors…"); }

    std::ostringstream cmd;
    cmd << "migraphx-driver compile"
        << " --onnx " << shellQuote(onnxPath.string())
        << " --gpu"
        << " --enable-offload-copy"
        << " --output " << shellQuote(mxrPath.string());
    if (compilePrecision == ModelPrecision::Fp16) {
        cmd << " --fp16";
    }

    std::string inputProbeError;
    bool usedDimParamFallback = false;
#ifdef AVE_HAVE_MIGRAPHX
    if (!appendDriverInputDims(cmd, onnxPath, compileWidth, compileHeight, inputProbeError)) {
        appendDriverDimParamFallback(cmd, compileWidth, compileHeight);
        usedDimParamFallback = true;
    }
#else
    appendDriverDimParamFallback(cmd, compileWidth, compileHeight);
    usedDimParamFallback = true;
#endif

    const char* timeoutEnv = std::getenv("AVE_MIGRAPHX_COMPILE_TIMEOUT_SEC");
    int timeoutSeconds = 45 * 60;
    if (timeoutEnv != nullptr) {
        try {
            timeoutSeconds = std::stoi(timeoutEnv);
        } catch (...) {
            timeoutSeconds = 45 * 60;
        }
    }
    if (timeoutSeconds < 60) { timeoutSeconds = 60; }

    const bool timeoutAvailable = commandInPath("timeout");
    std::ostringstream wrappedCmd;
    if (timeoutAvailable) {
        wrappedCmd << "timeout --foreground " << timeoutSeconds << "s " << cmd.str();
    } else {
        wrappedCmd << cmd.str();
    }
    const std::string cmdStr = wrappedCmd.str() + " 2>&1";

    if (progressCb && usedDimParamFallback) {
        progressCb(modelId, 0.03f,
            "ONNX input inspection failed; using symbolic dimension fallback…");
    }
    if (progressCb) { progressCb(modelId, 0.05f, "Launching migraphx-driver…"); }

    std::atomic<bool> compileDone{false};
    std::atomic<int> compileExit{-1};
    std::string lastOutputLine;
    std::string capturedOutput;
    std::mutex outputMtx;

    std::thread compileThread([&]() {
        FILE* pipe = popen(cmdStr.c_str(), "r");
        if (pipe == nullptr) {
            compileExit.store(-1);
            compileDone.store(true);
            return;
        }
        char buf[512];
        while (fgets(buf, sizeof(buf), pipe) != nullptr) {
            const std::string line = trimLine(std::string(buf));
            if (line.empty()) { continue; }
            std::lock_guard<std::mutex> lk(outputMtx);
            lastOutputLine = line;
            if (capturedOutput.size() < 16384) {
                capturedOutput.append(line);
                capturedOutput.push_back('\n');
            }
        }
        compileExit.store(pclose(pipe));
        compileDone.store(true);
    });

    int elapsedSec = 0;
    while (!compileDone.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (compileDone.load()) { break; }
        elapsedSec += 2;
        if (progressCb) {
            std::string line;
            {
                std::lock_guard<std::mutex> lk(outputMtx);
                line = lastOutputLine;
            }
            const int mins = elapsedSec / 60;
            const int secs = elapsedSec % 60;
            std::ostringstream status;
            status << "migraphx-driver compiling… "
                   << mins << "m " << secs << "s elapsed";
            if (!line.empty()) { status << " — " << line; }
            progressCb(modelId, -1.0f, status.str());
        }
    }
    compileThread.join();

    const int rawRc = compileExit.load();
    int exitCode = rawRc;
    int signalNum = 0;
#if defined(__unix__) || defined(__APPLE__)
    if (WIFEXITED(rawRc)) {
        exitCode = WEXITSTATUS(rawRc);
    } else if (WIFSIGNALED(rawRc)) {
        signalNum = WTERMSIG(rawRc);
        exitCode = 128 + signalNum;
    }
#endif

    if (exitCode != 0) {
        std::ostringstream msg;
        if (signalNum > 0) {
            msg << "migraphx-driver killed by signal " << signalNum;
            if (signalNum == 6) {
                msg << " (SIGABRT). This usually means MiGraphX hit an unsupported operator or internal assertion.";
            } else if (signalNum == 11) {
                msg << " (SIGSEGV).";
            } else {
                msg << '.';
            }
        } else if (timeoutAvailable && exitCode == 124) {
            msg << "migraphx-driver timed out after " << timeoutSeconds
                << " seconds. Set AVE_MIGRAPHX_COMPILE_TIMEOUT_SEC to a larger value if this model really needs longer.";
        } else if (rawRc == -1) {
            msg << "Failed to launch migraphx-driver.";
        } else {
            msg << "migraphx-driver exited with code " << exitCode << '.';
        }
        std::string driverOutput;
        {
            std::lock_guard<std::mutex> lk(outputMtx);
            driverOutput = capturedOutput.empty() ? lastOutputLine : capturedOutput;
        }
        if (!inputProbeError.empty()) {
            msg << "\nInput inspection note:\n" << inputProbeError;
        }
        if (!driverOutput.empty()) {
            msg << "\nDriver output:\n" << driverOutput;
        }
        error = msg.str();
        if (progressCb) {
            progressCb(modelId, 0.0f,
                "Compilation failed (exit " + std::to_string(exitCode) + ")");
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

// ─── MiGraphX compilation ─────────────────────────────────────────
// Uses the migraphx-driver command-line tool if present.
// The driver is shipped with ROCm and provides:
//   migraphx-driver compile --onnx <in.onnx> --output <out.mxr>
bool migraphxCompile(const std::filesystem::path& onnxPath,
                     const std::filesystem::path& mxrPath,
                     int compileWidth,
                     int compileHeight,
                     ModelPrecision compilePrecision,
                     std::optional<std::string> calibrationVideoPath,
                     const ModelProgressCb& progressCb,
                     const std::string& modelId,
                     std::string& error) {
    // ── Pre-flight: quantized-op scan ────────────────────────────
    // Run before compilation to detect quantized ops MiGraphX cannot lower.
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
                << "Action: use a non-quantized ONNX export before compiling with MiGraphX.";
            error = msg.str();
            if (progressCb) {
                progressCb(modelId, 0.0f,
                    "Unsupported: quantized model (" + quantOps[0] + " …)");
            }
            return false;
        }
    }

    if (compileWithMigraphxDriver(onnxPath, mxrPath, compileWidth, compileHeight,
                                  compilePrecision,
                                  progressCb, modelId, error)) {
        return true;
    }

#ifdef AVE_HAVE_MIGRAPHX
    const bool needsLibraryCompile =
        compilePrecision == ModelPrecision::Int8 ||
        error.find("migraphx-driver not found in PATH") != std::string::npos;
    if (!needsLibraryCompile) {
        return false;
    }

    if (progressCb && compilePrecision != ModelPrecision::Int8) {
        progressCb(modelId, 0.02f,
            "migraphx-driver not found; falling back to MiGraphX C++ runtime…");
    }
    return compileWithMigraphxLibrary(
        onnxPath, mxrPath, compileWidth, compileHeight, compilePrecision,
        calibrationVideoPath,
        progressCb, modelId, error);
#else
    return false;
#endif
}

// ─────────────────────────────────────────────────────────────────
// torchExportToOnnx — convert a TorchScript .pt / .pth model to ONNX
// using the ambient Python + torch environment (torch-MiGraphX stack).
//
// Writes a dynamic-axes ONNX (opset 17, NCHW 1×3×H×W) to onnxDest.
// Tries torch.jit.load() first; falls back to torch.load() for plain
// state-dict / nn.Module checkpoints.
// ─────────────────────────────────────────────────────────────────
bool torchExportToOnnx(const std::filesystem::path& pthPath,
                       const std::filesystem::path& onnxDest,
                       std::string& error) {
    if (!commandInPath("python3")) {
        error = "python3 not found in PATH; cannot convert PyTorch model. "
                "Install Python + torch: pip install torch";
        return false;
    }

    // Inline Python that handles both TorchScript and nn.Module checkpoints.
    const std::string pyScript =
        "import sys, torch\n"
        "pth, out = sys.argv[1], sys.argv[2]\n"
        "try:\n"
        "    model = torch.jit.load(pth, map_location='cpu')\n"
        "except Exception:\n"
        "    ckpt = torch.load(pth, map_location='cpu', weights_only=False)\n"
        "    model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt\n"
        "if hasattr(model, 'eval'): model.eval()\n"
        "dummy = torch.zeros(1, 3, 256, 256)\n"
        "torch.onnx.export(\n"
        "    model, dummy, out,\n"
        "    input_names=['input'], output_names=['output'],\n"
        "    dynamic_axes={'input':{0:'b',2:'h',3:'w'},'output':{0:'b',2:'h',3:'w'}},\n"
        "    opset_version=17)\n"
        "print('OK: exported to', out)\n";

    const auto scriptPath = std::filesystem::temp_directory_path()
                          / "ave_torch_export.py";
    {
        std::ofstream sf(scriptPath);
        if (!sf) {
            error = "Cannot write temp export script to " + scriptPath.string();
            return false;
        }
        sf << pyScript;
    }

    std::ostringstream cmd;
    cmd << "python3 " << scriptPath.string()
        << " \"" << pthPath.string()  << "\""
        << " \"" << onnxDest.string() << "\"";
    const int rc = std::system(cmd.str().c_str());

    std::error_code ec2;
    std::filesystem::remove(scriptPath, ec2);

    if (rc != 0) {
        error = "torch.onnx.export() failed (exit " + std::to_string(rc)
              + ").  Ensure torch is installed: pip install torch torch_migraphx";
        return false;
    }
    if (!fileExists(onnxDest)) {
        error = "Export ran but ONNX output not found: " + onnxDest.string();
        return false;
    }
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
            m.convertedPath.clear();
            const std::array<ModelPrecision, 3> preferredPrecisions = {
                ModelPrecision::Fp16,
                ModelPrecision::Int8,
                ModelPrecision::Fp32,
            };
            for (const auto precision : preferredPrecisions) {
                const auto p = compiledArtifactPath(convertedDir(), id, precision);
                if (fileExists(p)) {
                    m.convertedPath = p.string();
                    break;
                }
            }
        }

        // Derive state (do not overwrite transient states like Downloading)
        if (m.state == ModelState::Downloading ||
            m.state == ModelState::Converting) {
            return;
        }

        if (!m.convertedPath.empty()) {
            m.state = ModelState::Converted;
        } else if (!m.downloadedPath.empty() && hasMxrExtension(m.downloadedPath)) {
            // Downloaded .mxr models are inference-ready without a separate convert step.
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
ModelManager::ModelManager() : impl_(std::make_shared<Impl>()) {
    impl_->modelsDir = defaultModelsDir();
    ensureDir(impl_->downloadedDir());
    ensureDir(impl_->convertedDir());

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
            m.state == ModelState::Converted) {
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

    // Capture impl_ as a shared_ptr so the background thread keeps
    // the Impl alive even if ModelManager is destroyed before the
    // download thread finishes.
    std::shared_ptr<Impl> implPtr = impl_;
    std::thread([implPtr, entry, dlDir, progressCb, stateCb, modelId, cancelFlag]() mutable {
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
            std::lock_guard<std::mutex> lock(implPtr->mtx);
            auto& m = implPtr->records[modelId];
            if (ok) {
                // Clear the transient Downloading state before scanning so
                // scanAndUpdate() can derive the correct final state.
                m.state = ModelState::NotDownloaded;
                implPtr->scanAndUpdate(m);
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
                                      ModelPrecision         compilePrecision) {
    if (!isSupportedMiGraphXCompilePrecision(compilePrecision)) {
        error = "MiGraphX compilation currently supports fp32, fp16, and int8 artifacts; requested "
              + compilePrecisionTag(compilePrecision) + ".";
        return false;
    }
    if (compilePrecision == ModelPrecision::Int8) {
        error = "Generic MiGraphX int8 conversion is not supported because int8 compilation "
                "requires calibration frames from a real input video. "
                "Use runtime auto-compile with AVE_MIGRAPHX_PRECISION=int8 instead.";
        return false;
    }

    std::string modelPath;
    ModelFormat sourceFormat = ModelFormat::Onnx;
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
        if (m.entry.sourceFormat != ModelFormat::Onnx &&
            m.entry.sourceFormat != ModelFormat::Pytorch) {
            error = "MiGraphX compilation requires ONNX or PyTorch (.pth/.pt) format.";
            return false;
        }
        sourceFormat = m.entry.sourceFormat;
        modelPath    = m.downloadedPath;
        impl_->records[modelId].state = ModelState::Converting;
    }
    if (stateCb) { stateCb(modelId, ModelState::Converting); }

    // If the model is a PyTorch checkpoint, export it to a temporary ONNX via
    // torch-MiGraphX (torch.onnx.export, opset 17) before calling migraphx-driver.
    std::string onnxPath = modelPath;
    std::filesystem::path tempOnnxPath;
    if (sourceFormat == ModelFormat::Pytorch) {
        if (progressCb) { progressCb(modelId, 0.02f, "Exporting PyTorch model to ONNX…"); }
        tempOnnxPath = std::filesystem::temp_directory_path()
                     / (modelId + "_torch_export.onnx");
        std::string exportErr;
        if (!torchExportToOnnx(modelPath, tempOnnxPath, exportErr)) {
            error = "PyTorch → ONNX export failed: " + exportErr;
            ModelState finalState = ModelState::Error;
            {
                std::lock_guard<std::mutex> lockErr(impl_->mtx);
                auto& mErr        = impl_->records[modelId];
                mErr.state        = ModelState::Error;
                mErr.errorMessage = error;
                finalState        = mErr.state;
            }
            if (stateCb) { stateCb(modelId, finalState); }
            return false;
        }
        onnxPath = tempOnnxPath.string();
    }

    const int baselineWidth = baselineCompileWidth();
    const int baselineHeight = baselineCompileHeight();
    const auto mxrPath = compiledArtifactPath(impl_->convertedDir(), modelId, compilePrecision);
    bool ok = migraphxCompile(onnxPath, mxrPath,
                              baselineWidth, baselineHeight,
                              compilePrecision,
                              std::nullopt,
                              progressCb, modelId, error);
    if (!ok && shouldRetryFp32AfterTimeout(compilePrecision, error)) {
        const auto fp32Path = compiledArtifactPath(
            impl_->convertedDir(), modelId, ModelPrecision::Fp32);
        if (progressCb) {
            progressCb(modelId, 0.02f,
                "fp16 compilation timed out; retrying with fp32 artifact…");
        }
        if (fileExists(fp32Path)) {
            ok = true;
        } else {
            std::string fp32Error;
            ok = migraphxCompile(onnxPath, fp32Path,
                                 baselineWidth, baselineHeight,
                                 ModelPrecision::Fp32,
                                 std::nullopt,
                                 progressCb, modelId, fp32Error);
            if (!ok) {
                error += "\n\nfp32 fallback also failed:\n" + fp32Error;
            }
        }
        if (ok) {
            error.clear();
        }
    }

    ModelState finalState = ModelState::Error;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto& m = impl_->records[modelId];
        if (ok) {
            // Clear the transient Converting state before scanning so
            // scanAndUpdate() can derive the correct final state from
            // the files on disk (it early-returns for transient states).
            m.state = ModelState::Downloaded;
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

// ─── autoCompileForInference ───────────────────────────────────────
// Automatically compile a model for inference if not already compiled.
std::optional<std::string> ModelManager::autoCompileForInference(
    const std::string& modelId,
    std::string& error,
    std::optional<std::int64_t> inputWidth,
    std::optional<std::int64_t> inputHeight,
    ModelPrecision compilePrecision,
    std::optional<std::string> calibrationVideoPath) {

    if (!isSupportedMiGraphXCompilePrecision(compilePrecision)) {
        error = "MiGraphX auto-compile currently supports fp32, fp16, and int8 artifacts; requested "
              + compilePrecisionTag(compilePrecision) + ".";
        return std::nullopt;
    }

    const bool useCustomDims = inputWidth.has_value() || inputHeight.has_value();
    if (useCustomDims && (!inputWidth.has_value() || !inputHeight.has_value())) {
        error = "autoCompileForInference requires both inputWidth and inputHeight when either is provided.";
        return std::nullopt;
    }

    int compileWidth = baselineCompileWidth();
    int compileHeight = baselineCompileHeight();
    if (useCustomDims) {
        if (*inputWidth <= 0 || *inputHeight <= 0 ||
            *inputWidth > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
            *inputHeight > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            error = "Invalid compile dimensions: " + std::to_string(*inputWidth) + "x" + std::to_string(*inputHeight);
            return std::nullopt;
        }
        compileWidth = static_cast<int>(*inputWidth);
        compileHeight = static_cast<int>(*inputHeight);
    }

    auto customMxrPath = [&]() -> std::filesystem::path {
        return compiledArtifactPath(impl_->convertedDir(), modelId, compilePrecision,
                                    compileWidth, compileHeight);
    };
    auto customMxrPathForPrecision = [&](ModelPrecision precision) -> std::filesystem::path {
        return compiledArtifactPath(impl_->convertedDir(), modelId, precision,
                                    compileWidth, compileHeight);
    };

    if (useCustomDims) {
        const auto mxrPath = customMxrPath();
        if (fileExists(mxrPath)) {
            std::cout << "[auto-compile] Model '" << modelId << "' already compiled for "
                      << compileWidth << "x" << compileHeight << ": " << mxrPath << std::endl;
            return mxrPath.string();
        }
        if (compilePrecision == ModelPrecision::Fp16) {
            const auto fp32Path = customMxrPathForPrecision(ModelPrecision::Fp32);
            if (fileExists(fp32Path)) {
                std::cout << "[auto-compile] Reusing fp32 artifact for '" << modelId
                          << "' at " << compileWidth << "x" << compileHeight
                          << ": " << fp32Path << std::endl;
                return fp32Path.string();
            }
        }
    } else {
        // First check if already compiled (default path selection).
        {
            std::lock_guard<std::mutex> lock(impl_->mtx);
            auto it = impl_->records.find(modelId);
            if (it == impl_->records.end()) {
                error = "Unknown model id: " + modelId;
                return std::nullopt;
            }
            impl_->scanAndUpdate(it->second);
        }
        auto bestPath = bestPathForModel(modelId);
        if (bestPath && hasMxrExtension(*bestPath)) {
            std::cout << "[auto-compile] Model '" << modelId << "' already compiled: " << *bestPath << std::endl;
            return bestPath;
        }

        error = "[NeedsFrameSize] MiGraphX auto-compile requires actual input dimensions for the first compile of model '"
              + modelId + "'.";
        if (compilePrecision == ModelPrecision::Int8) {
            error += " Int8 also requires calibration frames from the real input video.";
        }
        error += " Deferring compile until the input video has been probed.";
        return std::nullopt;
    }

    if (compilePrecision == ModelPrecision::Int8 &&
        (!calibrationVideoPath.has_value() || calibrationVideoPath->empty())) {
        error = "MiGraphX int8 auto-compile requires a calibration video path.";
        return std::nullopt;
    }

    // Check if model is downloaded and capture source format/path.
    std::string modelPath;
    ModelFormat sourceFormat = ModelFormat::Onnx;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto it = impl_->records.find(modelId);
        if (it == impl_->records.end()) {
            error = "Unknown model id: " + modelId;
            return std::nullopt;
        }
        const auto& m = it->second;
        if (m.downloadedPath.empty() || m.downloadedPath == "(builtin)") {
            error = "Model not yet downloaded – cannot compile.";
            return std::nullopt;
        }
        if (m.entry.sourceFormat != ModelFormat::Onnx &&
            m.entry.sourceFormat != ModelFormat::Pytorch) {
            error = "MiGraphX compilation requires ONNX or PyTorch (.pth/.pt) format.";
            return std::nullopt;
        }
        modelPath = m.downloadedPath;
        sourceFormat = m.entry.sourceFormat;
    }

    // PyTorch checkpoints are exported to temporary ONNX before compile.
    std::string onnxPath = modelPath;
    std::filesystem::path tempOnnxPath;
    if (sourceFormat == ModelFormat::Pytorch) {
        tempOnnxPath = std::filesystem::temp_directory_path()
                     / (modelId + "_torch_export.onnx");
        std::string exportErr;
        if (!torchExportToOnnx(modelPath, tempOnnxPath, exportErr)) {
            error = "PyTorch → ONNX export failed: " + exportErr;
            return std::nullopt;
        }
        onnxPath = tempOnnxPath.string();
    }

    // Progress callback that logs to stdout
    auto progressCb = [](const std::string&, float progress, const std::string& msg) {
        std::cout << "[auto-compile] ";
        if (progress >= 0.0f) {
            std::cout << static_cast<int>(progress * 100) << "%: ";
        }
        std::cout << msg << std::endl;
    };

    std::cout << "[auto-compile] Compiling '" << modelId
              << "' at " << compileWidth << "x" << compileHeight << "..." << std::endl;

    if (useCustomDims) {
        const auto mxrPath = customMxrPath();
        if (migraphxCompile(onnxPath, mxrPath, compileWidth, compileHeight,
                            compilePrecision,
                            calibrationVideoPath,
                            progressCb, modelId, error)) {
            std::cout << "[auto-compile] Successfully compiled to: " << mxrPath << std::endl;
            return mxrPath.string();
        }
        if (shouldRetryFp32AfterTimeout(compilePrecision, error)) {
            const auto fp32Path = customMxrPathForPrecision(ModelPrecision::Fp32);
            std::cout << "[auto-compile] fp16 compilation timed out; retrying fp32 at "
                      << compileWidth << "x" << compileHeight << "..." << std::endl;
            if (fileExists(fp32Path)) {
                std::cout << "[auto-compile] Reusing fp32 artifact: " << fp32Path << std::endl;
                error.clear();
                return fp32Path.string();
            }
            std::string fp32Error;
            if (migraphxCompile(onnxPath, fp32Path, compileWidth, compileHeight,
                                ModelPrecision::Fp32,
                                calibrationVideoPath,
                                progressCb, modelId, fp32Error)) {
                std::cout << "[auto-compile] Successfully compiled to: " << fp32Path << std::endl;
                error.clear();
                return fp32Path.string();
            }
            error += "\n\nfp32 fallback also failed:\n" + fp32Error;
        }
        return std::nullopt;
    }

    if (convertToMiGraphX(modelId, progressCb, ModelStateCb{}, error, compilePrecision)) {
        refresh();
        auto bestPath = bestPathForModel(modelId);
        if (bestPath && hasMxrExtension(*bestPath)) {
            std::cout << "[auto-compile] Successfully compiled to: " << *bestPath << std::endl;
            return bestPath;
        }
    }

    if (error.empty()) {
        error = "Compilation failed: compiled .mxr was not found after convert step.";
    }
    return std::nullopt;
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

    // Sort: inference-ready first (converted > downloaded), then by name
    for (const auto& m : models) {
        DropdownEntry de;
        de.modelId       = m.entry.id;
        de.label         = statePrefix(m.state) + " " + m.entry.displayName;
        de.inferenceReady = (m.state == ModelState::Downloaded  ||
                             m.state == ModelState::Converted);
        out.push_back(de);
    }

    std::stable_sort(out.begin(), out.end(), [](const DropdownEntry& a, const DropdownEntry& b) {
        if (a.inferenceReady != b.inferenceReady) { return a.inferenceReady > b.inferenceReady; }
        return false;
    });

    return out;
}

}  // namespace ave
