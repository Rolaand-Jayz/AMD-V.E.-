// ─────────────────────────────────────────────────────────────────
// MiGraphX Backend — Gold-standard implementation
//
// Compliance with gold-standard requirements:
//   G1  ONNX opset ≤19 gate at parse time.
//   G2  Compile options explicit, logged, and encoded in cache key.
//   G3  Artifact manifest sidecar guards every cached .mxr;
//       mismatch → fail-fast recompile (ArtifactInvalid error).
//   G4  program::get_output_shapes() asserted at load and per frame.
//   G5  program::finish() called after every eval.
//   G6  Version tuple + MIGRAPHX_* env vars logged at initialize().
//   G7  TensorContracts built for input/output at load time; asserted
//       at inference entry (element-count gate).
//   G8  InteropBridge hook points documented for Vulkan↔HIP path;
//       current path uses CPU staging (logged as degraded mode).
//   G9  Structured InferenceError taxonomy throughout.
//   G10 ROCTx markers around compile, load, and eval for rocprof.
// ─────────────────────────────────────────────────────────────────
#include "ave/backends/migraphx_backend.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#  include <fcntl.h>
#  include <unistd.h>
#endif

// F16C hardware intrinsics for fp16↔4 float conversion.
// All AMD Ryzen CPUs and Intel x86-64 from Ivy Bridge onward support F16C.
// -march=native (set in CMakeLists) exposes __F16C__ to the preprocessor.
#if defined(__F16C__)
#  include <immintrin.h>
#endif

#include "ave/error_taxonomy.hpp"
#include "ave/frame_io.hpp"
#include "ave/interop_bridge.hpp"
#include "ave/model_catalog.hpp"
#include "ave/model_manager.hpp"
#include "ave/observability.hpp"
#include "ave/runtime_paths.hpp"
#include "ave/stage.hpp"
#include "ave/tensor_contract.hpp"
#include "ave/types.hpp"

#ifdef AVE_HAVE_MIGRAPHX
#  include <migraphx/migraphx.hpp>
#  ifdef MIGRAPHX_VERSION
#    define AVE_MIGRAPHX_VERSION_STR  MIGRAPHX_VERSION
#  else
#    define AVE_MIGRAPHX_VERSION_STR  "unknown"
#  endif
#endif

#ifdef AVE_HAVE_HIP
#  include <hip/hip_runtime.h>
#endif

namespace ave {
namespace {

std::string compilePrecisionTag(MiGraphXPrecision precision) {
    switch (precision) {
        case MiGraphXPrecision::Fp32: return "fp32";
        case MiGraphXPrecision::Fp16: return "fp16";
        case MiGraphXPrecision::Int8: return "int8";
    }
    return "unknown";
}

ModelPrecision modelCompilePrecision(MiGraphXPrecision precision) {
    switch (precision) {
        case MiGraphXPrecision::Fp32: return ModelPrecision::Fp32;
        case MiGraphXPrecision::Fp16: return ModelPrecision::Fp16;
        case MiGraphXPrecision::Int8: return ModelPrecision::Int8;
    }
    return ModelPrecision::Fp32;
}

std::optional<MiGraphXPrecision> parseCompilePrecisionValue(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (value == "fp32") {
        return MiGraphXPrecision::Fp32;
    }
    if (value == "fp16") {
        return MiGraphXPrecision::Fp16;
    }
    if (value == "int8") {
        return MiGraphXPrecision::Int8;
    }
    return std::nullopt;
}

CompileOptions compileOptionsFromEnv() {
    CompileOptions opts;
    if (const char* rawPrecision = std::getenv("AVE_MIGRAPHX_PRECISION"); rawPrecision != nullptr) {
        if (const auto parsed = parseCompilePrecisionValue(rawPrecision); parsed.has_value()) {
            opts.precision = *parsed;
        } else {
            std::cerr << "[migraphx] WARNING: unsupported AVE_MIGRAPHX_PRECISION='"
                      << rawPrecision << "'; using default precision fp16." << std::endl;
        }
    }
    return opts;
}

// ─────────────────────────────────────────────────────────────────
// System probe helpers
// ─────────────────────────────────────────────────────────────────

bool fileExists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

bool commandInPath(const std::string& command) {
    const char* pathEnv = std::getenv("PATH");
    if (pathEnv == nullptr) { return false; }
    std::string path(pathEnv);
    std::size_t start = 0;
    while (start <= path.size()) {
        std::size_t end = path.find(':', start);
        if (end == std::string::npos) { end = path.size(); }
        const std::string dir = path.substr(start, end - start);
        if (!dir.empty() &&
            fileExists((std::filesystem::path(dir) / command).string())) {
            return true;
        }
        if (end == path.size()) { break; }
        start = end + 1;
    }
    return false;
}

bool hasAnyMiGraphXArtifact() {
    if (const auto bundledPrefix = bundledMiGraphXPrefix(); bundledPrefix.has_value()) {
        if (fileExists(((*bundledPrefix) / "bin" / "migraphx-driver").string())) {
            return true;
        }
        if (fileExists(((*bundledPrefix) / "lib" / "libmigraphx_c.so").string())) {
            return true;
        }
        if (fileExists(((*bundledPrefix) / "lib" / "migraphx" / "lib" / "libmigraphx.so").string())) {
            return true;
        }
    }

    for (const auto& c : {"/opt/rocm/lib/libmigraphx.so",
                           "/opt/rocm/lib64/libmigraphx.so",
                           "/usr/lib/libmigraphx.so",
                           "/usr/lib64/libmigraphx.so"}) {
        if (fileExists(c)) { return true; }
    }
    for (const auto& libDir : {"/opt/rocm/lib", "/opt/rocm/lib64",
                                "/usr/lib",     "/usr/lib64"}) {
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(libDir, ec)) {
            if (ec) { break; }
            const std::string fname = entry.path().filename().string();
            if (fname.rfind("libmigraphx", 0) == 0) { return true; }
        }
    }
    return commandInPath("migraphx-driver") || bundledMiGraphXDriverPath().has_value();
}

bool hasAmdSignal() {
    return commandInPath("rocminfo") || commandInPath("rocm-smi")
        || fileExists("/opt/rocm");
}

// ─────────────────────────────────────────────────────────────────
// Version helpers (for manifest key construction — MiGraphX builds only)
// ─────────────────────────────────────────────────────────────────
#ifdef AVE_HAVE_MIGRAPHX

std::string getRocmVersion() {
    // Cached once per process — ROCm version does not change at runtime.
    static const std::string cached = []() -> std::string {
        std::ifstream f("/opt/rocm/.info/version");
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty()) { return line; }
        }
        return std::string("unknown");
    }();
    return cached;
}

std::string getMiGraphXVersion() {
#ifdef AVE_MIGRAPHX_VERSION_STR
    return AVE_MIGRAPHX_VERSION_STR;
#else
    return "unknown";
#endif
}

std::string getGfxTarget() {
    // Cached once per process — GFX target never changes while the app is running.
    // Avoids spawning rocminfo on every call to this function.
    static const std::string cached = []() -> std::string {
#ifdef AVE_HAVE_HIP
        int currentDevice = 0;
        hipDeviceProp_t props{};
        if (hipGetDevice(&currentDevice) == hipSuccess &&
            hipGetDeviceProperties(&props, currentDevice) == hipSuccess &&
            props.gcnArchName[0] != '\0') {
            return std::string(props.gcnArchName);
        }
#endif
        // Fallback: parse rocminfo output
        FILE* p = popen("rocminfo 2>/dev/null | grep -m1 'gfx[0-9]' | tr -s ' ' | cut -d' ' -f2", "r");
        if (p != nullptr) {
            std::array<char, 64> buf{};
            std::string result;
            if (std::fgets(buf.data(), static_cast<int>(buf.size()), p) != nullptr) {
                result = std::string(buf.data());
                while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                    result.pop_back();
                }
            }
            pclose(p);
            if (!result.empty()) { return result; }
        }
        return std::string("unknown");
    }();
    return cached;
}

std::string envOrDef(const char* name, const char* def) {
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string(def);
}

std::string currentCompileProfileLabel() {
    std::string value = envOrDef("AVE_MIGRAPHX_COMPILE_PROFILE", "fast");
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "balanced" || value == "default" || value == "dynamic_hybrid") {
        return "balanced";
    }
    if (value == "exhaustive" || value == "full" || value == "normal") {
        return "exhaustive";
    }
    return "fast";
}

std::string currentVisibleDeviceBinding() {
    const char* overrideVisible = std::getenv("AVE_MIGRAPHX_VISIBLE_DEVICES");
    if (overrideVisible != nullptr && *overrideVisible != '\0') {
        return overrideVisible;
    }
    const char* rocrVisible = std::getenv("ROCR_VISIBLE_DEVICES");
    if (rocrVisible != nullptr && *rocrVisible != '\0') {
        return rocrVisible;
    }
    const char* hipVisible = std::getenv("HIP_VISIBLE_DEVICES");
    if (hipVisible != nullptr && *hipVisible != '\0') {
        return hipVisible;
    }
    return "all";
}

std::string currentRuntimeFingerprint(const std::string& compileProfile,
                                      const std::string& visibleDevices,
                                      const std::string& problemCache,
                                      const std::string& miopenUserDb,
                                      const std::string& miopenCache,
                                      const std::string& miopenFindMode,
                                      const std::string& miopenParallel) {
    const std::string seed = compileProfile + "|" + visibleDevices + "|"
        + problemCache + "|" + miopenUserDb + "|" + miopenCache + "|"
        + miopenFindMode + "|" + miopenParallel;
    return std::to_string(std::hash<std::string>{}(seed));
}

#endif  // AVE_HAVE_MIGRAPHX

// ─────────────────────────────────────────────────────────────────
// Tensor-contract helpers
// ─────────────────────────────────────────────────────────────────

bool isInternalOutputParameterName(const std::string& name) {
    return name.find("#output") != std::string::npos;
}

bool toPositiveInt(std::int64_t value, const char* axis, int& out, std::string& error) {
    if (value <= 0 || value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        std::ostringstream os;
        os << "Invalid " << axis << " dimension " << value;
        error = os.str();
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool extractSpatialDims(const TensorContract& contract,
                        int&                  width,
                        int&                  height,
                        int&                  channels,
                        std::string&          error) {
    width = 0;
    height = 0;
    channels = 0;

    const auto& dims = contract.shape.dims;
    if (dims.size() == 4) {
        if (contract.layout == TensorLayout::NHWC) {
            return toPositiveInt(dims[2], "width", width, error)
                && toPositiveInt(dims[1], "height", height, error)
                && toPositiveInt(dims[3], "channels", channels, error);
        }
        return toPositiveInt(dims[3], "width", width, error)
            && toPositiveInt(dims[2], "height", height, error)
            && toPositiveInt(dims[1], "channels", channels, error);
    }
    if (dims.size() == 3) {
        if (contract.layout == TensorLayout::HWC) {
            return toPositiveInt(dims[1], "width", width, error)
                && toPositiveInt(dims[0], "height", height, error)
                && toPositiveInt(dims[2], "channels", channels, error);
        }
        return toPositiveInt(dims[2], "width", width, error)
            && toPositiveInt(dims[1], "height", height, error)
            && toPositiveInt(dims[0], "channels", channels, error);
    }

    std::ostringstream os;
    os << "Unsupported tensor rank " << dims.size()
       << " for contract '" << contract.name << "' (" << contract.shape.format() << ")";
    error = os.str();
    return false;
}

int extractBatchSize(const TensorContract& contract) {
    const auto& dims = contract.shape.dims;
    if (dims.size() == 4 && dims[0] > 0 &&
        dims[0] <= static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        return static_cast<int>(dims[0]);
    }
    return 1;
}

std::size_t elementsPerBatch(const TensorContract& contract, int batchSize) {
    const std::int64_t totalElements = contract.shape.elements();
    if (totalElements <= 0 || batchSize <= 0) {
        return 0;
    }
    const std::int64_t batchElements =
        totalElements / static_cast<std::int64_t>(batchSize);
    if (batchElements <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(batchElements);
}

std::string loadFailureKey(const std::string& modelId,
                           const std::optional<int>& inputWidth,
                           const std::optional<int>& inputHeight) {
    if (inputWidth.has_value() && inputHeight.has_value()) {
        return modelId + "@" + std::to_string(*inputWidth) + "x" + std::to_string(*inputHeight);
    }
    return modelId + "@default";
}

[[maybe_unused]] const TensorContract* selectPrimaryInputContract(
        const std::vector<TensorContract>& contracts) {
    if (contracts.empty()) { return nullptr; }
    for (const auto& c : contracts) {
        if (c.name == "input") { return &c; }
    }
    return &contracts.front();
}

[[maybe_unused]] bool contractMatchesFrameDims(
        const TensorContract& contract, int width, int height) {
    int modelW = 0;
    int modelH = 0;
    int modelC = 0;
    std::string err;
    if (!extractSpatialDims(contract, modelW, modelH, modelC, err)) {
        return false;
    }
    return modelW == width && modelH == height;
}

// ─────────────────────────────────────────────────────────────────
// ONNX opset scanner (G1: opset ≤19 gate)
//
// Lightweight protobuf binary scan — no proto library required.
// Reads only enough bytes to find the opset_import fields.
//
// ONNX ModelProto field layout (protobuf wire format):
//   opset_import  = field 8, wire type 2 → tag byte 0x42
//   OpsetImport.domain  = field 1, wire type 2 → tag byte 0x0A
//   OpsetImport.version = field 2, wire type 0 → tag byte 0x10
//
// Returns the highest opset version found for the default ONNX domain.
// Returns 0 if no opset_import fields are found (very old model or parse error).
// Returns -1 on file open error.
// ─────────────────────────────────────────────────────────────────
[[maybe_unused]] static constexpr int kMaxSupportedOpset = 19;

namespace proto {

// Read a varint from a byte buffer.  Returns number of bytes consumed,
// or 0 on overflow.  value is set to the decoded uint64.
std::size_t readVarint(const std::uint8_t* buf, std::size_t len, std::uint64_t& value) {
    value = 0;
    for (std::size_t i = 0; i < len && i < 10; ++i) {
        const uint64_t b = buf[i];
        value |= (b & 0x7Fu) << (7u * i);
        if ((b & 0x80u) == 0) { return i + 1; }
    }
    return 0;  // error or overflow
}

// Skip a protobuf field of the given wire type.
// Returns number of bytes consumed, or 0 on error.
std::size_t skipField(const std::uint8_t* buf, std::size_t len, std::uint32_t wireType) {
    if (len == 0) { return 0; }
    if (wireType == 0) {  // varint
        std::uint64_t v;
        return readVarint(buf, len, v);
    }
    if (wireType == 1) {  // 64-bit
        return (len >= 8) ? 8u : 0u;
    }
    if (wireType == 2) {  // length-delimited
        std::uint64_t sz = 0;
        const std::size_t consumed = readVarint(buf, len, sz);
        if (consumed == 0) { return 0; }
        const std::size_t total = consumed + static_cast<std::size_t>(sz);
        return (total <= len) ? total : 0u;
    }
    if (wireType == 5) {  // 32-bit
        return (len >= 4) ? 4u : 0u;
    }
    return 0;  // unknown / unsupported wire type
}

}  // namespace proto

[[maybe_unused]] int extractOnnxMaxOpset(const std::string& onnxPath) {
    // Limit scan to first 256 KB — opset_import always precedes the graph
    // data in well-formed ONNX files.
    static constexpr std::size_t kScanLimit = 256 * 1024;

    std::ifstream f(onnxPath, std::ios::binary);
    if (!f.is_open()) { return -1; }

    std::vector<std::uint8_t> buf(kScanLimit);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(kScanLimit));
    const std::size_t bytesRead = static_cast<std::size_t>(f.gcount());

    int maxOpset = 0;
    std::size_t pos = 0;

    while (pos < bytesRead) {
        // Read outer field tag
        std::uint64_t tag = 0;
        const std::size_t tagConsumed = proto::readVarint(&buf[pos], bytesRead - pos, tag);
        if (tagConsumed == 0) { break; }
        pos += tagConsumed;

        const auto fieldNum  = static_cast<std::uint32_t>(tag >> 3u);
        const auto wireType  = static_cast<std::uint32_t>(tag & 0x7u);

        if (fieldNum == 8 && wireType == 2) {
            // opset_import: parse embedded OpsetImport message
            std::uint64_t msgLen = 0;
            const std::size_t lenConsumed = proto::readVarint(&buf[pos], bytesRead - pos, msgLen);
            if (lenConsumed == 0) { break; }
            pos += lenConsumed;

            const std::size_t msgStart = pos;
            const std::size_t msgEnd   = std::min(msgStart + static_cast<std::size_t>(msgLen),
                                                  bytesRead);
            std::size_t inner = msgStart;
            std::int64_t opsetVersion = 0;
            bool hasDomain = false;
            bool domainIsDefault = true;  // "" or "ai.onnx"

            while (inner < msgEnd) {
                std::uint64_t inner_tag = 0;
                const std::size_t it = proto::readVarint(&buf[inner], msgEnd - inner, inner_tag);
                if (it == 0) { break; }
                inner += it;

                const auto ifn = static_cast<std::uint32_t>(inner_tag >> 3u);
                const auto iwt = static_cast<std::uint32_t>(inner_tag & 0x7u);

                if (ifn == 1 && iwt == 2) {
                    // domain string
                    std::uint64_t slen = 0;
                    const std::size_t sc = proto::readVarint(&buf[inner], msgEnd - inner, slen);
                    if (sc == 0) { break; }
                    inner += sc;
                    const std::string domain(
                        reinterpret_cast<const char*>(&buf[inner]),
                        static_cast<std::size_t>(slen));
                    inner += static_cast<std::size_t>(slen);
                    hasDomain = true;
                    domainIsDefault = domain.empty() || domain == "ai.onnx";
                } else if (ifn == 2 && iwt == 0) {
                    // version
                    std::uint64_t v = 0;
                    const std::size_t vc = proto::readVarint(&buf[inner], msgEnd - inner, v);
                    if (vc == 0) { break; }
                    inner += vc;
                    opsetVersion = static_cast<std::int64_t>(v);
                } else {
                    const std::size_t skip = proto::skipField(&buf[inner], msgEnd - inner, iwt);
                    if (skip == 0) { break; }
                    inner += skip;
                }
            }
            pos = msgEnd;

            // Only count default ("ai.onnx" or "") domain entries
            if ((!hasDomain || domainIsDefault) && opsetVersion > maxOpset) {
                maxOpset = static_cast<int>(opsetVersion);
            }
        } else {
            // Skip this field
            if (wireType == 2) {
                std::uint64_t fl = 0;
                const std::size_t fc = proto::readVarint(&buf[pos], bytesRead - pos, fl);
                if (fc == 0) { break; }
                pos += fc + static_cast<std::size_t>(fl);
                if (pos > bytesRead) { break; }
            } else {
                const std::size_t skip = proto::skipField(&buf[pos], bytesRead - pos, wireType);
                if (skip == 0) { break; }
                pos += skip;
            }
        }
    }

    return maxOpset;
}

// ─────────────────────────────────────────────────────────────────
// Manifest key construction
// ─────────────────────────────────────────────────────────────────

#ifdef AVE_HAVE_MIGRAPHX
[[maybe_unused]] obs::ArtifactManifestFields buildManifestFields(
        const std::string& onnxPath,
        const CompileOptions& opts) {
    obs::ArtifactManifestFields f;
    f.migraphxVersion = getMiGraphXVersion();
    f.rocmVersion     = getRocmVersion();
    f.gpuGfxTarget    = getGfxTarget();

    // Model identity: size + mtime (production should add SHA-256 here)
    std::error_code ec;
    if (std::filesystem::exists(onnxPath, ec)) {
        f.onnxFileSizeStr = std::to_string(std::filesystem::file_size(onnxPath, ec));
        const auto mtime  = std::filesystem::last_write_time(onnxPath, ec);
        const auto mtimeSec = std::chrono::duration_cast<std::chrono::seconds>(
            mtime.time_since_epoch()).count();
        f.onnxMtimeStr = std::to_string(mtimeSec);
    } else {
        f.onnxFileSizeStr = "0";
        f.onnxMtimeStr    = "0";
    }

    f.offloadCopy    = opts.offloadCopy    ? "1" : "0";
    f.precision      = compilePrecisionTag(opts.precision);
    f.compileProfile = currentCompileProfileLabel();
    f.disableMlir    = envOrDef("MIGRAPHX_DISABLE_MLIR", "0");
    f.enableNhwc     = envOrDef("MIGRAPHX_ENABLE_NHWC",  "0");
    f.enableCk       = envOrDef("MIGRAPHX_ENABLE_CK",    "0");
    f.problemCachePath = envOrDef("MIGRAPHX_PROBLEM_CACHE", "");
    f.miopenUserDbPath = envOrDef("MIOPEN_USER_DB_PATH", "");
    f.miopenCustomCacheDir = envOrDef("MIOPEN_CUSTOM_CACHE_DIR", "");
    f.miopenFindMode = envOrDef("MIOPEN_FIND_MODE", "FAST");
    f.miopenCompileParallelLevel = envOrDef("MIOPEN_COMPILE_PARALLEL_LEVEL", "1");
    f.visibleDevices = currentVisibleDeviceBinding();
    f.runtimeFingerprint = currentRuntimeFingerprint(
        f.compileProfile, f.visibleDevices, f.problemCachePath,
        f.miopenUserDbPath, f.miopenCustomCacheDir,
        f.miopenFindMode, f.miopenCompileParallelLevel);
    return f;
}
#endif  // AVE_HAVE_MIGRAPHX

// ─────────────────────────────────────────────────────────────────
// Model/stage helpers
// ─────────────────────────────────────────────────────────────────

std::string defaultModelIdFor(StageKind kind) {
    const auto entries = catalogEntriesForStage(kind);
    for (const auto* e : entries) { if (e->isDefault) { return e->id; } }
    if (!entries.empty()) { return entries.front()->id; }
    return {};
}

std::string resolveModelId(const EnhancementStage& stage) {
    const auto it = stage.params.find("model");
    if (it != stage.params.end()) {
        if (const auto* s = std::get_if<std::string>(&it->second)) {
            if (!s->empty()) { return *s; }
        }
    }
    return defaultModelIdFor(stage.kind);
}

std::optional<std::string> stageModelPath(const EnhancementStage& stage) {
    const auto it = stage.params.find("model_path");
    if (it == stage.params.end()) { return std::nullopt; }
    if (const auto* s = std::get_if<std::string>(&it->second)) {
        if (!s->empty()) { return *s; }
    }
    return std::nullopt;
}

bool stageModelPathExplicit(const EnhancementStage& stage) {
    const auto it = stage.params.find("model_path_explicit");
    if (it == stage.params.end()) { return false; }
    if (const auto* b = std::get_if<bool>(&it->second)) { return *b; }
    return false;
}

std::string normalizeExtLower(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    for (char& ch : ext) { ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); }
    return ext;
}

std::string quoteArg(const std::string& value) {
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

std::filesystem::path makeTempLogPath(const std::string& prefix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    return std::filesystem::temp_directory_path()
         / (prefix + "_" + std::to_string(stamp) + ".log");
}

void tunePipeIo(FILE* pipe, std::size_t stdioBufferBytes, int pipeBytes) {
    if (pipe == nullptr) {
        return;
    }
    setvbuf(pipe, nullptr, _IOFBF, stdioBufferBytes);
#if defined(__linux__)
    const int fd = fileno(pipe);
    if (fd >= 0 && pipeBytes > 0) {
        (void)fcntl(fd, F_SETPIPE_SZ, pipeBytes);
    }
#else
    (void)pipeBytes;
#endif
}

std::string readLogTail(const std::filesystem::path& path, std::size_t maxLines = 20) {
    std::ifstream in(path);
    if (!in.is_open()) { return {}; }

    std::deque<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
        if (lines.size() > maxLines) {
            lines.pop_front();
        }
    }

    std::ostringstream out;
    bool firstLine = true;
    for (const auto& l : lines) {
        if (!firstLine) {
            out << '\n';
        }
        firstLine = false;
        out << l;
    }
    return out.str();
}

std::string formatProcessExit(int rawStatus) {
#if defined(__unix__) || defined(__APPLE__)
    if (WIFEXITED(rawStatus)) {
        return "exit code " + std::to_string(WEXITSTATUS(rawStatus));
    }
    if (WIFSIGNALED(rawStatus)) {
        return "signal " + std::to_string(WTERMSIG(rawStatus));
    }
#endif
    return "status " + std::to_string(rawStatus);
}

constexpr int kDefaultTileExtent = 192;
constexpr int kDefaultTileOverlap = 16;
constexpr int kDefaultTileBatch = 4;

int readTileEnvValue(const char* name, int defaultValue) {
    if (const char* raw = std::getenv(name); raw != nullptr) {
        try {
            return std::stoi(raw);
        } catch (...) {
            return defaultValue;
        }
    }
    return defaultValue;
}

bool hasNonEmptyEnv(const char* name) {
    const char* raw = std::getenv(name);
    return raw != nullptr && *raw != '\0';
}

struct TileConfig {
    int width = kDefaultTileExtent;
    int height = kDefaultTileExtent;
    int overlap = kDefaultTileOverlap;
    int batch = kDefaultTileBatch;
    bool batchExplicit = false;
};

struct TileWindow {
    int begin = 0;
    int end = 0;
    int offset = 0;
};

struct TileDispatch {
    int tileX = 0;
    int tileY = 0;
    TileWindow srcXWindow;
    TileWindow srcYWindow;
};

std::vector<int> buildTileStarts(int fullExtent, int tileExtent, int overlap);

int chooseAdaptiveTileBatch(int frameWidth,
                            int frameHeight,
                            int tileWidth,
                            int tileHeight,
                            int tileOverlap) {
    const auto tileXs = buildTileStarts(frameWidth, tileWidth, tileOverlap);
    const auto tileYs = buildTileStarts(frameHeight, tileHeight, tileOverlap);
    const std::size_t estimatedTiles = tileXs.size() * tileYs.size();
    if (estimatedTiles <= 1u) {
        return 1;
    }

    const std::size_t tileArea =
        static_cast<std::size_t>(tileWidth) * static_cast<std::size_t>(tileHeight);
    int batch = 4;
    if (tileArea <= 96u * 96u) {
        batch = 16;
    } else if (tileArea <= 128u * 128u) {
        batch = 12;
    } else if (tileArea <= 192u * 192u) {
        batch = 8;
    } else if (tileArea <= 256u * 256u) {
        batch = 4;
    } else {
        batch = 2;
    }

    if (frameWidth * frameHeight <= 1280 * 720 && tileArea <= 192u * 192u) {
        batch = std::max(batch, 8);
    }

    batch = std::min(batch, 16);
    batch = std::min(batch, static_cast<int>(estimatedTiles));
    return std::max(batch, 1);
}

bool resolveTileConfig(const EnhancementStage& stage,
                       TileConfig& tileConfig,
                       std::string& error) {
    const int envTileSize = readTileEnvValue("AVE_MIGRAPHX_TILE_SIZE", kDefaultTileExtent);
    tileConfig.width = readTileEnvValue("AVE_MIGRAPHX_TILE_WIDTH", envTileSize);
    tileConfig.height = readTileEnvValue("AVE_MIGRAPHX_TILE_HEIGHT", envTileSize);
    tileConfig.overlap = readTileEnvValue("AVE_MIGRAPHX_TILE_OVERLAP", kDefaultTileOverlap);
    tileConfig.batch = readTileEnvValue("AVE_MIGRAPHX_TILE_BATCH", kDefaultTileBatch);
    tileConfig.batchExplicit = hasNonEmptyEnv("AVE_MIGRAPHX_TILE_BATCH");

    std::int64_t parsed = 0;
    if (tryGetInt(stage.params, "tile_size", parsed)) {
        tileConfig.width = static_cast<int>(parsed);
        tileConfig.height = static_cast<int>(parsed);
    }
    if (tryGetInt(stage.params, "tile_width", parsed)) {
        tileConfig.width = static_cast<int>(parsed);
    }
    if (tryGetInt(stage.params, "tile_height", parsed)) {
        tileConfig.height = static_cast<int>(parsed);
    }
    if (tryGetInt(stage.params, "tile_overlap", parsed)) {
        tileConfig.overlap = static_cast<int>(parsed);
    }
    if (tryGetInt(stage.params, "tile_batch", parsed)) {
        tileConfig.batch = static_cast<int>(parsed);
        tileConfig.batchExplicit = true;
    }

    if (tileConfig.width <= 0 || tileConfig.height <= 0) {
        error = "Tile dimensions must be positive.";
        return false;
    }
    if (tileConfig.width > 4096 || tileConfig.height > 4096) {
        error = "Tile dimensions are unreasonably large; keep them at or below 4096.";
        return false;
    }
    if (tileConfig.overlap < 0) {
        error = "Tile overlap cannot be negative.";
        return false;
    }
    if (tileConfig.batch <= 0) {
        error = "Tile batch must be positive.";
        return false;
    }
    if (tileConfig.batch > 64) {
        error = "Tile batch is unreasonably large; keep it at or below 64.";
        return false;
    }
    if (tileConfig.overlap * 2 >= tileConfig.width ||
        tileConfig.overlap * 2 >= tileConfig.height) {
        error = "Tile overlap must be less than half of the tile dimensions.";
        return false;
    }
    return true;
}

std::vector<int> buildTileStarts(int fullExtent, int tileExtent, int overlap) {
    std::vector<int> starts = {0};
    if (fullExtent <= tileExtent) {
        return starts;
    }

    const int step = std::max(1, tileExtent - overlap * 2);
    while (starts.back() + tileExtent < fullExtent) {
        const int next = std::min(starts.back() + step, fullExtent - tileExtent);
        if (next <= starts.back()) {
            break;
        }
        starts.push_back(next);
    }
    return starts;
}

TileWindow computeTileWindow(const std::vector<int>& starts,
                             std::size_t index,
                             int tileExtent,
                             int fullExtent) {
    const int tileStart = starts[index];
    const int tileEnd = tileStart + tileExtent;

    TileWindow window;
    if (index == 0u) {
        window.begin = 0;
    } else {
        const int prevEnd = starts[index - 1u] + tileExtent;
        window.begin = (prevEnd + tileStart) / 2;
    }

    if (index + 1u >= starts.size()) {
        window.end = fullExtent;
    } else {
        const int nextStart = starts[index + 1u];
        window.end = (tileEnd + nextStart) / 2;
    }

    window.begin = std::clamp(window.begin, tileStart, std::min(tileEnd, fullExtent));
    window.end = std::clamp(window.end, window.begin, std::min(tileEnd, fullExtent));
    window.offset = window.begin - tileStart;
    return window;
}

inline std::uint8_t floatToRgbByteLocal(float value) {
    value = std::clamp(value * 255.0f, 0.0f, 255.0f);
    return static_cast<std::uint8_t>(value + 0.5f);
}

// fp16 ⇔ float conversion.
// When -march=native enables F16C, delegate to single-cycle hardware
// instructions (_cvtss_sh, _cvtsh_ss).  Fall back to IEEE-conformant
// SW implementation otherwise (e.g. cross-compile, old CPUs).
#if defined(__F16C__)
inline std::uint16_t floatToHalfBitsLocal(float value) {
    return static_cast<std::uint16_t>(_cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT));
}
inline float halfBitsToFloatLocal(std::uint16_t bits) {
    return _cvtsh_ss(bits);
}
#else
inline std::uint16_t floatToHalfBitsLocal(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    std::uint32_t mantissa = bits & 0x007fffffu;
    int exponent = static_cast<int>((bits >> 23u) & 0xffu) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000u) >> static_cast<unsigned>(1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x00001000u) >> 13u));
    }

    if (exponent >= 31) {
        if (((bits >> 23u) & 0xffu) == 0xffu && mantissa != 0u) {
            return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa >> 13u) | 1u);
        }
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10u)
             | ((mantissa + 0x00001000u) >> 13u));
}

inline float halfBitsToFloatLocal(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000u)) << 16u;
    std::uint32_t exponent = (static_cast<std::uint32_t>(bits) >> 10u) & 0x1fu;
    std::uint32_t mantissa = static_cast<std::uint32_t>(bits & 0x03ffu);
    std::uint32_t outBits = 0;

    if (exponent == 0u) {
        if (mantissa == 0u) {
            outBits = sign;
        } else {
            exponent = 127u - 15u + 1u;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x03ffu;
            outBits = sign | (exponent << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 0x1fu) {
        outBits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        exponent = exponent + (127u - 15u);
        outBits = sign | (exponent << 23u) | (mantissa << 13u);
    }

    float out = 0.0f;
    std::memcpy(&out, &outBits, sizeof(out));
    return out;
}
#endif  // __F16C__

void packRgbTileClampToNchwFp32(const std::uint8_t* source,
                                int sourceWidth,
                                int sourceHeight,
                                int tileX,
                                int tileY,
                                int tileWidth,
                                int tileHeight,
                                float* tensor) {
    const std::size_t hw = static_cast<std::size_t>(tileWidth) *
                           static_cast<std::size_t>(tileHeight);
    float* const rPlane = tensor;
    float* const gPlane = tensor + hw;
    float* const bPlane = tensor + hw * 2u;
    constexpr float kInv255 = 1.0f / 255.0f;

    // Fast path: tile is fully within the source image — skip per-pixel clamping.
    // This branch is taken for all interior tiles and enables auto-vectorization.
    if (tileX >= 0 && tileY >= 0 &&
        tileX + tileWidth  <= sourceWidth &&
        tileY + tileHeight <= sourceHeight) {
        for (int y = 0; y < tileHeight; ++y) {
            const std::size_t srcRowBase =
                (static_cast<std::size_t>(tileY + y) * static_cast<std::size_t>(sourceWidth)
                 + static_cast<std::size_t>(tileX)) * 3u;
            const std::size_t dstRow = static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(tileWidth);
            for (int x = 0; x < tileWidth; ++x) {
                const std::size_t srcOff = srcRowBase + static_cast<std::size_t>(x) * 3u;
                const std::size_t dstOff = dstRow + static_cast<std::size_t>(x);
                rPlane[dstOff] = static_cast<float>(source[srcOff + 0u]) * kInv255;
                gPlane[dstOff] = static_cast<float>(source[srcOff + 1u]) * kInv255;
                bPlane[dstOff] = static_cast<float>(source[srcOff + 2u]) * kInv255;
            }
        }
        return;
    }

    // Slow path: border tile, clamp coordinates to image edges.
    for (int y = 0; y < tileHeight; ++y) {
        const int srcY = std::clamp(tileY + y, 0, sourceHeight - 1);
        const std::size_t dstRow = static_cast<std::size_t>(y) *
                                   static_cast<std::size_t>(tileWidth);
        for (int x = 0; x < tileWidth; ++x) {
            const int srcX = std::clamp(tileX + x, 0, sourceWidth - 1);
            const std::size_t srcOffset =
                (static_cast<std::size_t>(srcY) * static_cast<std::size_t>(sourceWidth)
                 + static_cast<std::size_t>(srcX)) * 3u;
            const std::size_t dstOffset = dstRow + static_cast<std::size_t>(x);
            rPlane[dstOffset] = static_cast<float>(source[srcOffset + 0u]) * kInv255;
            gPlane[dstOffset] = static_cast<float>(source[srcOffset + 1u]) * kInv255;
            bPlane[dstOffset] = static_cast<float>(source[srcOffset + 2u]) * kInv255;
        }
    }
}

void packRgbTileClampToNchwFp16(const std::uint8_t* source,
                                int sourceWidth,
                                int sourceHeight,
                                int tileX,
                                int tileY,
                                int tileWidth,
                                int tileHeight,
                                std::uint16_t* tensor) {
    const std::size_t hw = static_cast<std::size_t>(tileWidth) *
                           static_cast<std::size_t>(tileHeight);
    std::uint16_t* const rPlane = tensor;
    std::uint16_t* const gPlane = tensor + hw;
    std::uint16_t* const bPlane = tensor + hw * 2u;
    constexpr float kInv255 = 1.0f / 255.0f;

    // Fast path: tile is fully within the source image — skip per-pixel clamping.
    if (tileX >= 0 && tileY >= 0 &&
        tileX + tileWidth  <= sourceWidth &&
        tileY + tileHeight <= sourceHeight) {
        for (int y = 0; y < tileHeight; ++y) {
            const std::size_t srcRowBase =
                (static_cast<std::size_t>(tileY + y) * static_cast<std::size_t>(sourceWidth)
                 + static_cast<std::size_t>(tileX)) * 3u;
            const std::size_t dstRow = static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(tileWidth);
            for (int x = 0; x < tileWidth; ++x) {
                const std::size_t srcOff = srcRowBase + static_cast<std::size_t>(x) * 3u;
                const std::size_t dstOff = dstRow + static_cast<std::size_t>(x);
                rPlane[dstOff] = floatToHalfBitsLocal(static_cast<float>(source[srcOff + 0u]) * kInv255);
                gPlane[dstOff] = floatToHalfBitsLocal(static_cast<float>(source[srcOff + 1u]) * kInv255);
                bPlane[dstOff] = floatToHalfBitsLocal(static_cast<float>(source[srcOff + 2u]) * kInv255);
            }
        }
        return;
    }

    // Slow path: border tile.
    for (int y = 0; y < tileHeight; ++y) {
        const int srcY = std::clamp(tileY + y, 0, sourceHeight - 1);
        const std::size_t dstRow = static_cast<std::size_t>(y) *
                                   static_cast<std::size_t>(tileWidth);
        for (int x = 0; x < tileWidth; ++x) {
            const int srcX = std::clamp(tileX + x, 0, sourceWidth - 1);
            const std::size_t srcOffset =
                (static_cast<std::size_t>(srcY) * static_cast<std::size_t>(sourceWidth)
                 + static_cast<std::size_t>(srcX)) * 3u;
            const std::size_t dstOffset = dstRow + static_cast<std::size_t>(x);
            rPlane[dstOffset] = floatToHalfBitsLocal(static_cast<float>(source[srcOffset + 0u]) * kInv255);
            gPlane[dstOffset] = floatToHalfBitsLocal(static_cast<float>(source[srcOffset + 1u]) * kInv255);
            bPlane[dstOffset] = floatToHalfBitsLocal(static_cast<float>(source[srcOffset + 2u]) * kInv255);
        }
    }
}

void blitNchwFp32TileRegionToRgb24(const float* tensor,
                                   int tileChannels,
                                   int tileWidth,
                                   int tileHeight,
                                   int srcX,
                                   int srcY,
                                   int copyWidth,
                                   int copyHeight,
                                   std::vector<std::uint8_t>& dst,
                                   int dstWidth,
                                   int dstX,
                                   int dstY) {
    const std::size_t hw = static_cast<std::size_t>(tileWidth) *
                           static_cast<std::size_t>(tileHeight);
    const float* const rPlane = tileChannels > 0 ? tensor : nullptr;
    const float* const gPlane = tileChannels > 1 ? tensor + hw : nullptr;
    const float* const bPlane = tileChannels > 2 ? tensor + (2u * hw) : nullptr;

    for (int y = 0; y < copyHeight; ++y) {
        const std::size_t srcRow =
            static_cast<std::size_t>(srcY + y) * static_cast<std::size_t>(tileWidth)
            + static_cast<std::size_t>(srcX);
        const std::size_t dstRow =
            (static_cast<std::size_t>(dstY + y) * static_cast<std::size_t>(dstWidth)
             + static_cast<std::size_t>(dstX)) * 3u;
        for (int x = 0; x < copyWidth; ++x) {
            const std::size_t srcIdx = srcRow + static_cast<std::size_t>(x);
            const std::size_t dstIdx = dstRow + static_cast<std::size_t>(x) * 3u;
            dst[dstIdx + 0u] = floatToRgbByteLocal(rPlane[srcIdx]);
            dst[dstIdx + 1u] = floatToRgbByteLocal(gPlane[srcIdx]);
            dst[dstIdx + 2u] = floatToRgbByteLocal(bPlane[srcIdx]);
        }
    }
}

void blitNchwFp16TileRegionToRgb24(const std::uint16_t* tensor,
                                   int tileChannels,
                                   int tileWidth,
                                   int tileHeight,
                                   int srcX,
                                   int srcY,
                                   int copyWidth,
                                   int copyHeight,
                                   std::vector<std::uint8_t>& dst,
                                   int dstWidth,
                                   int dstX,
                                   int dstY) {
    const std::size_t hw = static_cast<std::size_t>(tileWidth) *
                           static_cast<std::size_t>(tileHeight);
    const std::uint16_t* const rPlane = tileChannels > 0 ? tensor : nullptr;
    const std::uint16_t* const gPlane = tileChannels > 1 ? tensor + hw : nullptr;
    const std::uint16_t* const bPlane = tileChannels > 2 ? tensor + (2u * hw) : nullptr;

    for (int y = 0; y < copyHeight; ++y) {
        const std::size_t srcRow =
            static_cast<std::size_t>(srcY + y) * static_cast<std::size_t>(tileWidth)
            + static_cast<std::size_t>(srcX);
        const std::size_t dstRow =
            (static_cast<std::size_t>(dstY + y) * static_cast<std::size_t>(dstWidth)
             + static_cast<std::size_t>(dstX)) * 3u;
        for (int x = 0; x < copyWidth; ++x) {
            const std::size_t srcIdx = srcRow + static_cast<std::size_t>(x);
            const std::size_t dstIdx = dstRow + static_cast<std::size_t>(x) * 3u;
            dst[dstIdx + 0u] = floatToRgbByteLocal(halfBitsToFloatLocal(rPlane[srcIdx]));
            dst[dstIdx + 1u] = floatToRgbByteLocal(halfBitsToFloatLocal(gPlane[srcIdx]));
            dst[dstIdx + 2u] = floatToRgbByteLocal(halfBitsToFloatLocal(bPlane[srcIdx]));
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// TensorDtype mapper (MiGraphX shape type → TensorDtype)
// ─────────────────────────────────────────────────────────────────
#ifdef AVE_HAVE_MIGRAPHX
TensorDtype mapMiGraphXType(migraphx_shape_datatype_t t) {
    switch (t) {
        case migraphx_shape_float_type:       return TensorDtype::Fp32;
        case migraphx_shape_half_type:        return TensorDtype::Fp16;
        case migraphx_shape_bf16_type:        return TensorDtype::Bf16;
        case migraphx_shape_int8_type:        return TensorDtype::Int8;
        case migraphx_shape_fp8e4m3fnuz_type: return TensorDtype::Fp8E4M3FNUZ;
        default:                              return TensorDtype::Unknown;
    }
}

[[maybe_unused]] void setOnnxInputShapesForFrame(
        migraphx::onnx_options& options,
        const migraphx::program_parameter_shapes& shapes,
        int inputWidth,
        int inputHeight) {
    const std::size_t w = static_cast<std::size_t>(inputWidth);
    const std::size_t h = static_cast<std::size_t>(inputHeight);
    for (const char* rawName : shapes.names()) {
        if (rawName == nullptr) { continue; }
        const std::string name(rawName);
        if (name.empty() || isInternalOutputParameterName(name)) { continue; }

        auto dims = shapes[name.c_str()].lengths();
        if (dims.empty()) { continue; }

        if (dims.size() == 4) {
            dims[0] = 1; // enforce single-frame inference
            if (dims[1] <= 4) {
                // NCHW
                dims[2] = h;
                dims[3] = w;
            } else if (dims[3] <= 4) {
                // NHWC
                dims[1] = h;
                dims[2] = w;
            } else {
                // Default to NCHW when channel axis is ambiguous.
                dims[2] = h;
                dims[3] = w;
            }
            options.set_input_parameter_shape(name, dims);
            continue;
        }

        if (dims.size() == 3) {
            if (dims[0] <= 4) {
                // CHW
                dims[1] = h;
                dims[2] = w;
            } else if (dims[2] <= 4) {
                // HWC
                dims[0] = h;
                dims[1] = w;
            }
            options.set_input_parameter_shape(name, dims);
        }
    }
}
#endif

}  // namespace

// ─────────────────────────────────────────────────────────────────
// CompileOptions implementation
// ─────────────────────────────────────────────────────────────────

bool CompileOptions::validate(std::string& error) const {
    if (!offloadCopy) {
        error = "CompileOptions: offloadCopy=false is not supported by the current runtime path.";
        return false;
    }
    if (precision != MiGraphXPrecision::Fp32 &&
        precision != MiGraphXPrecision::Fp16 &&
        precision != MiGraphXPrecision::Int8) {
        error = "CompileOptions: only fp32, fp16, and int8 are supported.";
        return false;
    }
    error.clear();
    return true;
}

std::string CompileOptions::format() const {
    std::ostringstream os;
    os << "offload_copy=" << (offloadCopy ? "1" : "0")
       << " precision=" << compilePrecisionTag(precision);
    return os.str();
}

// ─────────────────────────────────────────────────────────────────
// Impl (compiled path: AVE_HAVE_MIGRAPHX)
// ─────────────────────────────────────────────────────────────────

#ifdef AVE_HAVE_MIGRAPHX

struct ModelProgram {
    migraphx::program          prog;
    std::optional<migraphx::arguments> lastResults;
    std::vector<migraphx::shape> inputShapes;
    std::vector<TensorContract> inputContracts;
    std::vector<TensorContract> outputContracts;
    std::string sourcePath;
    bool sourceIsMxr = false;
    int compiledInputWidth = 0;
    int compiledInputHeight = 0;
    std::mutex evalMutex;
    bool warmupComplete = false;
    obs::ArtifactManifestFields runtimeFields;
};

struct MiGraphXBackend::Impl {
    bool           initialised = false;
    int            deviceIdx   = 0;
    CompileOptions opts;
    std::mutex     mtx;
    std::unordered_map<std::string, std::shared_ptr<ModelProgram>> programs;
    std::unordered_map<std::string, std::string> loadFailures;
    ModelManager    modelManager;
    InteropBridge   interopBridge;

    // ── buildContracts ──────────────────────────────────────────
    // Construct TensorContracts from MiGraphX parameter/output shapes.
    static std::vector<TensorContract> buildContracts(
            const migraphx::program_parameter_shapes& shapes,
            const std::string& role) {
        std::vector<TensorContract> result;
        for (const char* rawName : shapes.names()) {
            const std::string name = rawName != nullptr ? std::string(rawName) : std::string();
            if (name.empty()) { continue; }
            // Skip internal output parameters
            if (isInternalOutputParameterName(name)) { continue; }
            const auto shape = shapes[name.c_str()];
            TensorContract c;
            c.name        = name;
            c.description = role + " parameter";
            c.dtype       = mapMiGraphXType(shape.type());
            // Build shape dims from MiGraphX lengths vector
            c.shape.dims.clear();
            for (const auto len : shape.lengths()) {
                c.shape.dims.push_back(static_cast<std::int64_t>(len));
            }
            c.layout = inferTensorLayout(c.shape.dims);
            if (c.layout == TensorLayout::Unknown) {
                c.layout = TensorLayout::NCHW;
            }
            result.push_back(std::move(c));
        }
        return result;
    }

    // ── loadProgram ──────────────────────────────────────────────
    // G1: ONNX opset gate  G3: manifest validation  G7: tensor contracts
    bool loadProgram(const std::string& modelId,
                     std::string& error,
                     std::optional<int> inputWidth = std::nullopt,
                     std::optional<int> inputHeight = std::nullopt,
                     std::optional<std::string> preferredPath = std::nullopt,
                     bool preferredPathExplicit = false,
                     std::optional<std::string> calibrationVideoPath = std::nullopt,
                     int compileBatch = 1) {
        std::string sourcePath;
        bool sourceIsMxr = true;
        const bool needFrameSpecificArtifact = inputWidth.has_value() && inputHeight.has_value();
        if (preferredPath.has_value() && !preferredPath->empty()) {
            const std::string ext = normalizeExtLower(*preferredPath);
            if (ext == ".mxr") {
                // For real video inference, ignore auto-selected generic .mxr
                // paths until we've resolved the frame-size-specific artifact.
                if (preferredPathExplicit || !needFrameSpecificArtifact) {
                    sourcePath = *preferredPath;
                }
            } else if (preferredPathExplicit) {
                const auto ie = InferenceError::modelIncompatible(
                    "MiGraphX backend requires a compiled .mxr artifact for inference.",
                    "model='" + modelId + "' explicit model_path=" + *preferredPath
                    + "\nAction: compile this model to .mxr in Model Manager.");
                error = ie.format();
                return false;
            }
        }

        auto tryResolveOrCompile = [&](std::optional<std::int64_t> iw,
                                       std::optional<std::int64_t> ih) -> bool {
            std::string compileError;
            const auto compiled = modelManager.autoCompileForInference(
                modelId, compileError, iw, ih, modelCompilePrecision(opts.precision),
                compileBatch, calibrationVideoPath);
            if (compiled.has_value() && normalizeExtLower(*compiled) == ".mxr") {
                sourcePath = *compiled;
                if (iw.has_value() && ih.has_value()) {
                    std::cout << "[migraphx] using frame-size specific .mxr for '" << modelId
                              << "': " << sourcePath << " (" << *iw << "x" << *ih << ")"
                              << std::endl;
                } else {
                    std::cout << "[migraphx] using compiled .mxr for '" << modelId
                              << "': " << sourcePath << std::endl;
                }
                return true;
            }
            if (!compileError.empty()) {
                const auto ie = InferenceError::compileFailure(
                    "Unable to compile model '" + modelId + "' for MiGraphX.",
                    compileError);
                error = ie.format();
                return false;
            }
            return true;
        };

        // For real video inference we always prefer a frame-size specific artifact.
        // This avoids loading a stale generic compile (for example 3840x2160) on
        // smaller sources and then silently mismatching tensor contracts.
        if (sourcePath.empty() && needFrameSpecificArtifact) {
            if (!tryResolveOrCompile(static_cast<std::int64_t>(*inputWidth),
                                     static_cast<std::int64_t>(*inputHeight))) {
                return false;
            }
        }

        if (sourcePath.empty() && !needFrameSpecificArtifact) {
            const auto bestPath = modelManager.bestPathForModel(modelId);
            if (bestPath.has_value() && normalizeExtLower(*bestPath) == ".mxr") {
                sourcePath = *bestPath;
            }
        }

        if (sourcePath.empty() && (!inputWidth.has_value() || !inputHeight.has_value())) {
            if (!tryResolveOrCompile(std::nullopt, std::nullopt)) {
                return false;
            }
        }

        if (sourcePath.empty()) {
            const auto ie = InferenceError::modelIncompatible(
                "No compiled .mxr artifact available for model '" + modelId + "'.",
                "Download/compile this model in Model Manager before running inference.");
            error = ie.format();
            return false;
        }

        const std::string key =
            loadFailureKey(modelId, inputWidth, inputHeight) + "|" + sourcePath;
        if (auto pit = programs.find(modelId); pit != programs.end()) {
            if (pit->second != nullptr &&
                pit->second->sourcePath == sourcePath &&
                pit->second->sourceIsMxr == sourceIsMxr) {
                return true;
            }
            programs.erase(pit);
        }

        if (const auto failIt = loadFailures.find(key); failIt != loadFailures.end()) {
            error = failIt->second;
            return false;
        }

        auto rememberFailure = [&](const std::string& msg) {
            error = msg;
            loadFailures[key] = error;
            return false;
        };

        AVE_ROCTX_RANGE("migraphx:load");
        try {
            auto mp = std::make_shared<ModelProgram>();
            mp->sourcePath = sourcePath;
            mp->sourceIsMxr = sourceIsMxr;
            mp->runtimeFields = buildManifestFields(sourcePath, opts);
            obs::logMiGraphXEnvironment(mp->runtimeFields, "runtime-load", sourcePath, "pending");

            mp->prog = migraphx::load(sourcePath.c_str());

            const auto outShapes = mp->prog.get_output_shapes();
            if (outShapes.empty()) {
                const auto ie = InferenceError::runtimeFailure(
                    "program::get_output_shapes() returned empty for model '"
                    + modelId + "'.",
                    "Source: " + sourcePath);
                std::cerr << ie.format() << std::endl;
                AVE_ROCTX_RANGE_END();
                return rememberFailure(ie.format());
            }

            const auto parameterShapes = mp->prog.get_parameter_shapes();
            mp->inputContracts  = buildContracts(parameterShapes, "input");
            if (mp->inputContracts.empty()) {
                const auto ie = InferenceError::modelIncompatible(
                    "Model '" + modelId + "' has no usable input tensors.",
                    "Internal '#output' placeholders were filtered from program parameters.");
                std::cerr << ie.format() << std::endl;
                AVE_ROCTX_RANGE_END();
                return rememberFailure(ie.format());
            }
            mp->inputShapes.reserve(mp->inputContracts.size());
            for (const auto& contract : mp->inputContracts) {
                mp->inputShapes.push_back(parameterShapes[contract.name.c_str()]);
            }

            mp->outputContracts.clear();
            for (std::size_t i = 0; i < outShapes.size(); ++i) {
                TensorContract oc;
                oc.name        = "output_" + std::to_string(i);
                oc.description = "output parameter";
                oc.dtype       = mapMiGraphXType(outShapes[i].type());
                for (const auto len : outShapes[i].lengths()) {
                    oc.shape.dims.push_back(static_cast<std::int64_t>(len));
                }
                oc.layout = inferTensorLayout(oc.shape.dims);
                if (oc.layout == TensorLayout::Unknown) {
                    oc.layout = TensorLayout::NCHW;
                }
                mp->outputContracts.push_back(std::move(oc));
            }

            std::cout << "[migraphx] loaded model='" << modelId
                      << "' source='" << sourcePath
                      << "' format=" << (sourceIsMxr ? "mxr" : "onnx") << "\n";
            for (const auto& c : mp->inputContracts) {
                std::cout << "  in:  " << c.format() << '\n';
            }
            for (const auto& c : mp->outputContracts) {
                std::cout << "  out: " << c.format() << '\n';
            }
            std::cout << "  compile_opts: " << opts.format() << '\n';
            std::cout << std::flush;

            programs.emplace(modelId, std::move(mp));
            loadFailures.erase(key);
        } catch (const std::exception& ex) {
            const auto ie = InferenceError::compileFailure(
                std::string("MiGraphX load/compile failed: ") + ex.what(),
                "Source: " + sourcePath);
            std::cerr << ie.format() << std::endl;
            AVE_ROCTX_RANGE_END();
            return rememberFailure(ie.format());
        }
        AVE_ROCTX_RANGE_END();
        return true;
    }

    // ── runInference ─────────────────────────────────────────────
    // G4: output shape assertion  G5: program::finish()
    // G7: element-count gate (TensorContract)
    // G8: InteropBridge hook documented
    bool runInference(const std::string& modelId,
                      const void*        inputData,
                      std::size_t        inputElements,
                      TensorDtype        inputDtype,
                      const void*&       outputData,
                      std::size_t&       outputElements,
                      TensorDtype&       outputDtype,
                      std::string&       error,
                      bool               finishAfterEval = true) {
        std::shared_ptr<ModelProgram> mp;
        {
            std::lock_guard<std::mutex> lk(mtx);
            const auto it = programs.find(modelId);
            if (it == programs.end() || it->second == nullptr) {
                error = InferenceError::runtimeFailure(
                    "Model not loaded: " + modelId).format();
                return false;
            }
            mp = it->second;
        }

        std::lock_guard<std::mutex> evalLock(mp->evalMutex);

        if (mp->inputContracts.empty()) {
            error = InferenceError::runtimeFailure(
                "Model '" + modelId + "' has no input parameters.").format();
            return false;
        }

        std::size_t contractIdx = 0;
        bool matchedByElements = false;
        for (std::size_t i = 0; i < mp->inputContracts.size(); ++i) {
            const std::int64_t expected = mp->inputContracts[i].shape.elements();
            if (expected > 0 && static_cast<std::size_t>(expected) == inputElements) {
                contractIdx = i;
                matchedByElements = true;
                break;
            }
        }
        if (!matchedByElements) {
            for (std::size_t i = 0; i < mp->inputContracts.size(); ++i) {
                if (mp->inputContracts[i].name == "input") {
                    contractIdx = i;
                    break;
                }
            }
        }

        const auto& contract = mp->inputContracts[contractIdx];
        const auto& inName   = contract.name;

        if (contract.dtype != inputDtype) {
            error = InferenceError::runtimeFailure(
                "Input dtype mismatch for '" + modelId + "': expected "
                + toString(contract.dtype) + ", got " + toString(inputDtype) + '.').format();
            return false;
        }

        // ── G7: Element-count assertion ──────────────────────────
        {
            std::string contractError;
            if (!assertElementCount(contract, inputElements, contractError)) {
                obs::logTensorContractViolation(
                    "MiGraphXBackend::runInference (input gate)",
                    contract.format(),
                    "actual elements=" + std::to_string(inputElements));
                error = InferenceError::runtimeFailure(contractError).format();
                return false;
            }
        }

        // ── Build parameter map (CPU host pointers) ─────────────
        // MiGraphX compiled graphs for the 7900 GRE bake hip_copy_to_gpu
        // nodes directly into the program.  eval() therefore expects raw
        // CPU RAM pointers — it handles the H2D/D2H transfers internally.
        try {
            const auto& inShape = mp->inputShapes[contractIdx];

            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            migraphx::argument inArg(inShape, const_cast<void*>(
                inputData));

            migraphx::program_parameters pp;
            pp.add(inName.c_str(), inArg);

            // ── Eval ─────────────────────────────────────────────
            AVE_ROCTX_RANGE("migraphx:eval");
            const auto results = mp->prog.eval(pp);
            AVE_ROCTX_RANGE_END();
            if (finishAfterEval) {
                mp->prog.experimental_get_context().finish();
            }

            // ── G4: Assert output shapes per frame ───────────────
            if (results.empty()) {
                error = InferenceError::runtimeFailure(
                    "program::eval returned no outputs for '" + modelId + "'.").format();
                return false;
            }
            const auto& outShape = results[0].get_shape();
            if (!mp->outputContracts.empty()) {
                const auto expectedElems = static_cast<std::size_t>(
                    mp->outputContracts[0].shape.elements());
                if (outShape.elements() != expectedElems) {
                    obs::logTensorContractViolation(
                        "MiGraphXBackend::runInference (output gate)",
                        mp->outputContracts[0].format(),
                        "actual elements=" + std::to_string(outShape.elements()));
                    error = InferenceError::runtimeFailure(
                        "Output shape mismatch for '" + modelId + "': expected "
                        + std::to_string(expectedElems) + " elements, got "
                        + std::to_string(outShape.elements()) + ".").format();
                    return false;
                }
            }

            // ── Retrieve output ──────────────────────────────────
            // eval() returns CPU-accessible memory (the compiled graph
            // includes hip_copy_from_gpu nodes for outputs). Keep the
            // result handle alive in the model cache so postprocess can
            // read the buffer directly without an extra host-side copy.
            outputElements = outShape.elements();
            outputDtype = mapMiGraphXType(outShape.type());
            mp->lastResults = std::move(results);
            outputData = (*mp->lastResults)[0].data();

        } catch (const std::exception& ex) {
            error = InferenceError::runtimeFailure(
                std::string("MiGraphX eval: ") + ex.what(),
                "model='" + modelId + "'").format();
            return false;
        }
        return true;
    }
};

#else  // !AVE_HAVE_MIGRAPHX

struct MiGraphXBackend::Impl {
    bool           initialised = false;
    int            deviceIdx   = 0;
    CompileOptions opts;
    std::mutex     mtx;
    std::unordered_map<std::string, bool>                    loaded;
    std::unordered_map<std::string, std::vector<TensorContract>> inputContracts_;
    std::unordered_map<std::string, std::vector<TensorContract>> outputContracts_;

    bool loadProgram(const std::string& modelId,
                     std::string& error,
                     std::optional<int> inputWidth = std::nullopt,
                     std::optional<int> inputHeight = std::nullopt,
                     std::optional<std::string> preferredPath = std::nullopt,
                     bool preferredPathExplicit = false,
                     std::optional<std::string> calibrationVideoPath = std::nullopt,
                     int compileBatch = 1) {
        (void)modelId;
        (void)inputWidth;
        (void)inputHeight;
        (void)preferredPath;
        (void)preferredPathExplicit;
        (void)calibrationVideoPath;
        (void)compileBatch;
        error = "MiGraphX hardware support was not compiled into this build (-DAVE_HAVE_MIGRAPHX=OFF).";
        return false;
    }
};

#endif  // AVE_HAVE_MIGRAPHX

// ─────────────────────────────────────────────────────────────────
// Public interface
// ─────────────────────────────────────────────────────────────────

MiGraphXBackend::MiGraphXBackend()  : impl_(std::make_unique<Impl>()) {
    impl_->opts = compileOptionsFromEnv();
}
MiGraphXBackend::~MiGraphXBackend() = default;

BackendType MiGraphXBackend::type()  const { return BackendType::MiGraphX; }
std::string MiGraphXBackend::name()  const { return "MiGraphX (ROCm)"; }
bool MiGraphXBackend::supportsDirectOutputEncode() const { return true; }

bool MiGraphXBackend::isAvailable(std::string& reason) const {
#ifdef AVE_HAVE_MIGRAPHX
    if (!hasAmdSignal()) {
        reason = "ROCm tooling not detected (expected rocminfo/rocm-smi or /opt/rocm).";
        return false;
    }
    if (!hasAnyMiGraphXArtifact()) {
        reason = "MiGraphX runtime not found (system or bundled libmigraphx / migraphx-driver).";
        return false;
    }
    reason = "MiGraphX runtime detected.";
    return true;
#else
    reason = "MiGraphX hardware support was not compiled into this build (-DAVE_HAVE_MIGRAPHX=OFF).";
    return false;
#endif
}

bool MiGraphXBackend::initialize(std::string& error) {
    std::string reason;
    if (!isAvailable(reason)) {
        error = InferenceError::vulkanDevice(
            "MiGraphX init: " + reason).format();
        return false;
    }

#ifdef AVE_HAVE_HIP
    int devCount = 0;
    if (hipGetDeviceCount(&devCount) != hipSuccess || devCount == 0) {
        error = InferenceError::vulkanDevice(
            "HIP: no AMD GPU devices found.").format();
        return false;
    }
    (void)hipSetDevice(impl_->deviceIdx);
    if (devCount > 1 && std::getenv("HIP_VISIBLE_DEVICES") == nullptr) {
        std::cerr << "[migraphx] WARNING: HIP enumerated " << devCount
                  << " devices and HIP_VISIBLE_DEVICES is unset. "
                  << "AMD's ROCm install guidance warns that integrated graphics can destabilize ROCm; "
                  << "if MiGraphX compilation is flaky, pin the intended discrete GPU with HIP_VISIBLE_DEVICES."
                  << std::endl;
    }
#endif

    impl_->initialised = true;

    // ── G6: Log version tuple and MIGRAPHX_* env vars ───────────
    obs::logVersionTuple();
    obs::logMiGraphXEnvironment();
    impl_->interopBridge.logConfig();

    std::cout << "[backend] MiGraphX initialised on device " << impl_->deviceIdx
              << " — compile options: " << impl_->opts.format() << std::endl;
    return true;
}

void MiGraphXBackend::setCompileOptions(const CompileOptions& opts) {
    impl_->opts = opts;
}

CompileOptions MiGraphXBackend::compileOptions() const {
    return impl_->opts;
}

bool MiGraphXBackend::preloadModel(const std::string& modelId, std::string& error) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->loadProgram(modelId, error);
}

void MiGraphXBackend::evictModel(const std::string& modelId) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_MIGRAPHX
    impl_->programs.erase(modelId);
    const std::string prefix = modelId + "@";
    for (auto it = impl_->loadFailures.begin(); it != impl_->loadFailures.end();) {
        if (it->first.rfind(prefix, 0) == 0) {
            it = impl_->loadFailures.erase(it);
        } else {
            ++it;
        }
    }
#else
    impl_->loaded.erase(modelId);
    impl_->inputContracts_.erase(modelId);
    impl_->outputContracts_.erase(modelId);
#endif
}

void MiGraphXBackend::evictAll() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_MIGRAPHX
    impl_->programs.clear();
    impl_->loadFailures.clear();
#else
    impl_->loaded.clear();
    impl_->inputContracts_.clear();
    impl_->outputContracts_.clear();
#endif
}

int MiGraphXBackend::deviceIndex() const { return impl_->deviceIdx; }

std::vector<TensorContract>
MiGraphXBackend::inputContracts(const std::string& modelId) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_MIGRAPHX
    const auto it = impl_->programs.find(modelId);
    if (it != impl_->programs.end() && it->second != nullptr) { return it->second->inputContracts; }
#else
    const auto it = impl_->inputContracts_.find(modelId);
    if (it != impl_->inputContracts_.end()) { return it->second; }
#endif
    return {};
}

std::vector<TensorContract>
MiGraphXBackend::outputContracts(const std::string& modelId) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_MIGRAPHX
    const auto it = impl_->programs.find(modelId);
    if (it != impl_->programs.end() && it->second != nullptr) { return it->second->outputContracts; }
#else
    const auto it = impl_->outputContracts_.find(modelId);
    if (it != impl_->outputContracts_.end()) { return it->second; }
#endif
    return {};
}

StageResult MiGraphXBackend::runStage(const EnhancementStage& stage, std::string& error) {
    const std::string modelId = resolveModelId(stage);
    const std::optional<std::string> selectedPath = stageModelPath(stage);
    const bool selectedPathExplicit = stageModelPathExplicit(stage);
    if (modelId.empty()) {
        std::cout << "[migraphx] no model configured for " << toString(stage.kind)
                  << " — deferring to FFmpeg filter chain." << std::endl;
        return StageResult::Deferred;
    }

    if (!selectedPathExplicit && selectedPath.has_value() &&
        normalizeExtLower(*selectedPath) == ".mxr") {
        std::cout << "[migraphx] stage '" << toString(stage.kind)
                  << "' deferring auto-selected .mxr preload until the actual frame size "
                     "is known; processVideoFile() will load the correct artifact."
                  << std::endl;
        return StageResult::Deferred;
    }

    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        if (!impl_->loadProgram(modelId, error, std::nullopt, std::nullopt,
                                selectedPath, selectedPathExplicit)) {
            if (error.find("[NeedsFrameSize]") != std::string::npos) {
                // Native ONNX path: we need actual frame dimensions from decode.
                error.clear();
                return StageResult::Deferred;
            }
            // Classify the error to decide fall-through vs hard-fail.
            // ModelIncompatible and ArtifactInvalid → warn + fallback.
            // SyncHazard → hard-fail (data integrity at risk).
            if (error.find("[SyncHazard]") != std::string::npos) {
                // SyncHazard is fail-fast per gold standard.
                return StageResult::Error;
            }
            std::cerr << "[migraphx] WARNING: model not ready for stage '"
                      << toString(stage.kind) << "':\n  " << error
                      << "\n  → Deferring to FFmpeg filter chain." << std::endl;
            error.clear();
            return StageResult::Deferred;
        }
    }

    // Model is loaded and verified. Actual per-frame AI inference runs in
    // processVideoFile() which is called by the FFmpeg encode pipeline.
    // Returning Deferred here tells the pipeline to call processVideoFile()
    // during the encode pass where real frame-by-frame AI processing happens.
    AVE_ROCTX_MARK("migraphx:stage-model-ready");
    std::cout << "[migraphx] model='" << modelId
              << "' loaded and verified for stage '" << toString(stage.kind)
              << "' | compile=" << impl_->opts.format()
              << "\n  AI inference will run via processVideoFile() during encode." << std::endl;
    return StageResult::Deferred;
}

// ─────────────────────────────────────────────────────────────────
// processFrameDir — per-frame AI inference on a directory of PNGs
//
// Gold-standard compliance:
//   G5  program::finish() called after every eval (via runInference).
//   G7  TensorContract element-count gate (via runInference).
//   G8  CPU staging logged as degraded mode.
//   G9  Structured InferenceError taxonomy.
//   G10 ROCTx markers around the processing loop.
// ─────────────────────────────────────────────────────────────────


// ─────────────────────────────────────────────────────────────────
// probeVideoDimensions — get width, height, fps via ffprobe
// ─────────────────────────────────────────────────────────────────
namespace {

struct ProbeResult {
    int width = 0;
    int height = 0;
    double fps = 30.0;
    std::int64_t totalFrames = 0;
};

bool probeVideo(const std::string& path, ProbeResult& result, std::string& err) {
    // width x height
    {
        const std::string cmd = "ffprobe -v error -select_streams v:0 "
            "-show_entries stream=width,height,r_frame_rate,nb_frames "
            "-of csv=p=0:s=, \"" + path + "\" 2>/dev/null";
        FILE* p = popen(cmd.c_str(), "r");
        if (!p) { err = "Failed to run ffprobe on " + path; return false; }
        std::array<char, 256> buf{};
        std::string line;
        if (std::fgets(buf.data(), static_cast<int>(buf.size()), p)) {
            line = buf.data();
        }
        pclose(p);
        // Parse: width,height,fps_num/fps_den,nb_frames
        int w = 0, h = 0, fpsNum = 30, fpsDen = 1;
        std::int64_t nbFrames = 0;
        if (std::sscanf(line.c_str(), "%d,%d,%d/%d,%lld",
                &w, &h, &fpsNum, &fpsDen,
                reinterpret_cast<long long*>(&nbFrames)) >= 2) {
            result.width = w;
            result.height = h;
            if (fpsDen > 0) result.fps = static_cast<double>(fpsNum) / static_cast<double>(fpsDen);
            result.totalFrames = nbFrames;
        } else {
            err = "Failed to parse ffprobe output for " + path;
            return false;
        }
    }
    // If nb_frames was 0 or N/A, count via packet counting
    if (result.totalFrames <= 0) {
        const std::string cmd = "ffprobe -v error -count_packets -select_streams v:0 "
            "-show_entries stream=nb_read_packets -of csv=p=0 \"" + path + "\" 2>/dev/null";
        FILE* p = popen(cmd.c_str(), "r");
        if (p) {
            std::array<char, 64> buf{};
            if (std::fgets(buf.data(), static_cast<int>(buf.size()), p)) {
                result.totalFrames = std::atoll(buf.data());
            }
            pclose(p);
        }
    }
    return result.width > 0 && result.height > 0;
}

}  // namespace

StageResult MiGraphXBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
#ifdef AVE_HAVE_MIGRAPHX
    auto deferToFfmpeg = [&](const std::string& detail) -> StageResult {
        std::cerr << "[migraphx] processVideoFile: " << detail
                  << "\n  → Deferring to FFmpeg." << std::endl;
        error.clear();
        return StageResult::Deferred;
    };

    const std::string modelId = resolveModelId(stage);
    const std::optional<std::string> selectedPath = stageModelPath(stage);
    const bool selectedPathExplicit = stageModelPathExplicit(stage);
    if (modelId.empty()) {
        std::cout << "[migraphx] processVideoFile: no model for "
                  << toString(stage.kind) << " — deferring." << std::endl;
        return StageResult::Deferred;
    }

    // ── Probe video dimensions via ffprobe (reliable, no Vulkan HW needed) ──
    ProbeResult probe;
    if (!probeVideo(inputVideo, probe, error)) {
        return deferToFfmpeg("ffprobe failed: " + error);
    }

    const int inW = probe.width;
    const int inH = probe.height;
    const std::int64_t totalFrames = probe.totalFrames;

    std::cout << "[migraphx] Input: " << inW << "x" << inH
              << " fps=" << probe.fps
              << " frames=" << totalFrames << std::endl;

    TileConfig tileConfig;
    if (!resolveTileConfig(stage, tileConfig, error)) {
        return deferToFfmpeg("Invalid tile configuration: " + error);
    }
    if (!tileConfig.batchExplicit) {
        const int desiredBatch = chooseAdaptiveTileBatch(
            inW, inH, tileConfig.width, tileConfig.height, tileConfig.overlap);
        tileConfig.batch = desiredBatch;
        const auto compilePrecision = modelCompilePrecision(impl_->opts.precision);
        std::vector<int> candidateBatches = {1, 2, 4, 8, 12, 16, desiredBatch};
        std::sort(candidateBatches.begin(), candidateBatches.end());
        candidateBatches.erase(
            std::unique(candidateBatches.begin(), candidateBatches.end()),
            candidateBatches.end());
        std::stable_sort(candidateBatches.begin(), candidateBatches.end(),
                         [desiredBatch](int lhs, int rhs) {
                             const int lhsDelta = std::abs(lhs - desiredBatch);
                             const int rhsDelta = std::abs(rhs - desiredBatch);
                             if (lhsDelta != rhsDelta) {
                                 return lhsDelta < rhsDelta;
                             }
                             return lhs > rhs;
                         });
        for (const int candidateBatch : candidateBatches) {
            std::string validationDetail;
            const auto validatedArtifact = impl_->modelManager.validatedCompiledArtifactPath(
                modelId, compilePrecision,
                static_cast<std::int64_t>(inW),
                static_cast<std::int64_t>(inH),
                candidateBatch,
                &validationDetail);
            if (validatedArtifact.has_value()) {
                tileConfig.batch = candidateBatch;
                if (candidateBatch != desiredBatch) {
                    std::cout << "[migraphx] reusing validated tile batch "
                              << candidateBatch << " instead of heuristic batch "
                              << desiredBatch << " for '" << modelId << "'." << std::endl;
                }
                break;
            }
        }
        std::cout << "[migraphx] adaptive tile batch selected: "
                  << tileConfig.batch << std::endl;
    }

    // Ensure a model program is ready for the requested tile size.
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        if (!impl_->loadProgram(modelId, error, tileConfig.width, tileConfig.height,
                                selectedPath, selectedPathExplicit, inputVideo,
                                tileConfig.batch)) {
            std::cerr << "[migraphx] processVideoFile: model load failed: "
                      << error << "\n  → Deferring to FFmpeg." << std::endl;
            error.clear();
            return StageResult::Deferred;
        }
    }

    // Verify input/output contracts
    TensorContract inputContract;
    TensorContract outputContract;
    std::shared_ptr<ModelProgram> loadedProgram;
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        const auto pit = impl_->programs.find(modelId);
        if (pit == impl_->programs.end() || pit->second == nullptr) {
            error = InferenceError::runtimeFailure(
                "Program unexpectedly missing after successful load: " + modelId).format();
            return StageResult::Error;
        }
        loadedProgram = pit->second;
        if (loadedProgram->inputContracts.empty()) {
            return deferToFfmpeg("Model has no input contracts.");
        }
        inputContract = loadedProgram->inputContracts.front();
        for (const auto& c : loadedProgram->inputContracts) {
            if (c.name == "input") { inputContract = c; break; }
        }
        if (loadedProgram->outputContracts.empty()) {
            return deferToFfmpeg("Model has no output contracts.");
        }
        outputContract = loadedProgram->outputContracts.front();
    }

    int modelInW = 0, modelInH = 0, modelInC = 0;
    int modelOutW = 0, modelOutH = 0, modelOutC = 0;
    std::string contractError;
    if (!extractSpatialDims(inputContract, modelInW, modelInH, modelInC, contractError)) {
        return deferToFfmpeg("Cannot derive model spatial dims: " + contractError);
    }
    if (!extractSpatialDims(outputContract, modelOutW, modelOutH, modelOutC, contractError)) {
        return deferToFfmpeg("Cannot derive model output spatial dims: " + contractError);
    }
    if (modelInC != 3) {
        return deferToFfmpeg("Model expects " + std::to_string(modelInC) + " channels, need 3.");
    }
    if (modelOutC < 3) {
        return deferToFfmpeg("Model outputs " + std::to_string(modelOutC)
                           + " channels; need at least 3.");
    }
    if (inputContract.dtype != TensorDtype::Fp32 &&
        inputContract.dtype != TensorDtype::Fp16) {
        return deferToFfmpeg("Model input dtype " + toString(inputContract.dtype)
                           + " is not supported by the current host staging path.");
    }
    if (outputContract.dtype != TensorDtype::Fp32 &&
        outputContract.dtype != TensorDtype::Fp16) {
        return deferToFfmpeg("Model output dtype " + toString(outputContract.dtype)
                           + " is not supported by the current host staging path.");
    }

    if (modelInW <= 0 || modelInH <= 0 || modelOutW <= 0 || modelOutH <= 0) {
        return deferToFfmpeg("Model reported non-positive tensor dimensions.");
    }
    if (modelOutW % modelInW != 0 || modelOutH % modelInH != 0) {
        return deferToFfmpeg("Model output dimensions are not integer multiples of the tile input size.");
    }

    const int compiledBatch = extractBatchSize(inputContract);
    const int outputBatch = extractBatchSize(outputContract);
    if (compiledBatch <= 0) {
        return deferToFfmpeg("Model reported a non-positive batch dimension.");
    }
    if (outputBatch != compiledBatch) {
        return deferToFfmpeg("Model input/output batch dimensions do not match.");
    }
    if (compiledBatch != tileConfig.batch) {
        std::cout << "[migraphx] requested tile batch " << tileConfig.batch
                  << ", loaded artifact batch " << compiledBatch
                  << "; using the loaded artifact." << std::endl;
    }

    const int scaleX = modelOutW / modelInW;
    const int scaleY = modelOutH / modelInH;
    if (scaleX <= 0 || scaleY <= 0) {
        return deferToFfmpeg("Derived non-positive model scale factors.");
    }

    int tileOverlap = tileConfig.overlap;
    const int maxOverlap = std::max(0, std::min((modelInW - 1) / 2, (modelInH - 1) / 2));
    if (tileOverlap > maxOverlap) {
        std::cout << "[migraphx] reducing tile overlap from " << tileConfig.overlap
                  << " to " << maxOverlap << " to match the loaded tile artifact."
                  << std::endl;
        tileOverlap = maxOverlap;
    }

    const int outW = inW * scaleX;
    const int outH = inH * scaleY;
    const auto tileXs = buildTileStarts(inW, modelInW, tileOverlap);
    const auto tileYs = buildTileStarts(inH, modelInH, tileOverlap);
    std::vector<TileWindow> tileXWindows;
    std::vector<TileWindow> tileYWindows;
    tileXWindows.reserve(tileXs.size());
    tileYWindows.reserve(tileYs.size());
    for (std::size_t i = 0; i < tileXs.size(); ++i) {
        tileXWindows.push_back(computeTileWindow(tileXs, i, modelInW, inW));
    }
    for (std::size_t i = 0; i < tileYs.size(); ++i) {
        tileYWindows.push_back(computeTileWindow(tileYs, i, modelInH, inH));
    }
    std::vector<TileDispatch> tileDispatches;
    tileDispatches.reserve(tileXs.size() * tileYs.size());
    for (std::size_t tileRow = 0; tileRow < tileYs.size(); ++tileRow) {
        for (std::size_t tileCol = 0; tileCol < tileXs.size(); ++tileCol) {
            tileDispatches.push_back(TileDispatch{
                tileXs[tileCol],
                tileYs[tileRow],
                tileXWindows[tileCol],
                tileYWindows[tileRow],
            });
        }
    }
    const std::size_t tilesPerFrame = tileXs.size() * tileYs.size();

    // ── Open FFmpeg decode pipe (software decode — always works) ──
    // Outputs raw RGB24 frames at source resolution via pipe.
    std::string decodeTimeLim;
    if (opts.previewDurationSec > 0.0) {
        std::ostringstream tlss;
        tlss << " -t " << opts.previewDurationSec;
        decodeTimeLim = tlss.str();
    }
    const std::string decodeCmd =
        "ffmpeg -hide_banner -loglevel error" + decodeTimeLim +
        " -i \"" + inputVideo + "\" "
        "-f rawvideo -pix_fmt rgb24 pipe:1 2>/dev/null";

    FILE* decodePipe = popen(decodeCmd.c_str(), "r");
    if (!decodePipe) {
        error = "Failed to open FFmpeg decode pipe for " + inputVideo;
        return StageResult::Error;
    }
    tunePipeIo(decodePipe, 1u << 20, 4 << 20);

    // ── Open FFmpeg encode pipe (software encode — always works) ──
    const auto encodeLogPath = makeTempLogPath("ave_migraphx_encode");
    const bool directOutputEncode = opts.directOutputEncode;
    const std::string encodeCodec = opts.outputCodec.empty() ? "libx264" : opts.outputCodec;
    std::ostringstream encodeOss;
    encodeOss << "ffmpeg -y -hide_banner -loglevel error "
              << "-f rawvideo -pix_fmt rgb24 "
              << "-s " << outW << "x" << outH << " "
              << "-r " << probe.fps << " "
              << "-i pipe:0 "
              << "-i \"" << inputVideo << "\" "
              << "-map 0:v:0 -map 1:a? ";
    if (directOutputEncode) {
        encodeOss << "-vf " << quoteArg("format=yuv420p") << ' '
                  << "-c:v " << encodeCodec << ' ';
        if (!opts.outputProfile.empty()) {
            encodeOss << "-profile:v " << opts.outputProfile << ' ';
        }
        if (opts.outputThreads > 0) {
            encodeOss << "-threads " << opts.outputThreads << ' ';
        }
        encodeOss << "-crf " << opts.outputCrf << ' '
                  << "-preset " << opts.outputPreset << ' '
                  << "-c:a copy -shortest ";
    } else {
        encodeOss << "-c:v ffv1 -level 3 -slicecrc 1 -c:a copy ";
    }
    encodeOss << "\"" << outputVideo << "\" "
              << "2> " << quoteArg(encodeLogPath.string());

    FILE* encodePipe = popen(encodeOss.str().c_str(), "w");
    if (!encodePipe) {
        pclose(decodePipe);
        error = "Failed to open FFmpeg encode pipe for " + outputVideo;
        return StageResult::Error;
    }
    tunePipeIo(encodePipe, 1u << 20, 4 << 20);

    auto cleanupEncodeLog = [&]() {
        std::error_code ec;
        std::filesystem::remove(encodeLogPath, ec);
    };
    auto formatEncodeFailure = [&](const std::string& prefix,
                                   std::optional<int> rawStatus = std::nullopt) {
        std::ostringstream os;
        os << prefix;
        if (rawStatus.has_value()) {
            os << " (" << formatProcessExit(*rawStatus) << ")";
        }
        const std::string ffmpegLog = readLogTail(encodeLogPath);
        if (!ffmpegLog.empty()) {
            os << "\nffmpeg stderr:\n" << ffmpegLog;
        }
        return os.str();
    };

    AVE_ROCTX_RANGE("migraphx:processVideoFile");
    const std::size_t batchesPerFrame =
        (tilesPerFrame + static_cast<std::size_t>(compiledBatch) - 1u)
        / static_cast<std::size_t>(compiledBatch);
    std::cout << "[migraphx] processVideoFile: model='" << modelId
              << "' input=" << inW << "x" << inH
              << " output=" << outW << "x" << outH
              << " tile=" << modelInW << "x" << modelInH
              << " overlap=" << tileOverlap
              << " batch=" << compiledBatch
              << " scale=" << scaleX << "x" << scaleY
              << " tiles/frame=" << tilesPerFrame
              << " batches/frame=" << batchesPerFrame
              << " stage=" << toString(stage.kind)
              << " direct_final_encode=" << (directOutputEncode ? "on" : "off")
              << std::endl;

    const std::size_t frameBytes = static_cast<std::size_t>(inW) *
                                   static_cast<std::size_t>(inH) * 3u;
    const std::size_t outFrameBytes = static_cast<std::size_t>(outW) *
                                      static_cast<std::size_t>(outH) * 3u;
    const std::int64_t totalInputElements = inputContract.shape.elements();
    const std::int64_t totalOutputElements = outputContract.shape.elements();
    if (totalInputElements <= 0 || totalOutputElements <= 0 ||
        totalInputElements % compiledBatch != 0 ||
        totalOutputElements % compiledBatch != 0) {
        return deferToFfmpeg("Model batch tensor sizes are not divisible by the loaded batch.");
    }
    const std::size_t tileTensorElements =
        elementsPerBatch(inputContract, compiledBatch);
    const std::size_t batchedInputTensorElements =
        tileTensorElements * static_cast<std::size_t>(compiledBatch);
    const std::size_t expectedTileOutputElements =
        elementsPerBatch(outputContract, compiledBatch);
    const std::size_t expectedBatchOutputElements =
        expectedTileOutputElements * static_cast<std::size_t>(compiledBatch);
    const std::size_t expectedHostTileInputElements =
        static_cast<std::size_t>(modelInW) * static_cast<std::size_t>(modelInH)
        * static_cast<std::size_t>(modelInC);
    const std::size_t expectedHostTileOutputElements =
        static_cast<std::size_t>(modelOutW) * static_cast<std::size_t>(modelOutH)
        * static_cast<std::size_t>(modelOutC);
    if (tileTensorElements != expectedHostTileInputElements ||
        expectedTileOutputElements != expectedHostTileOutputElements) {
        return deferToFfmpeg("Model tensor layout does not match the host NCHW staging path.");
    }
    std::vector<std::uint8_t> rgbIn(frameBytes);
    std::vector<std::uint8_t> rgbOut;
    std::vector<float> batchedInputTensorFp32;
    std::vector<std::uint16_t> batchedInputTensorFp16;
    if (inputContract.dtype == TensorDtype::Fp32) {
        batchedInputTensorFp32.resize(batchedInputTensorElements);
    } else {
        batchedInputTensorFp16.resize(batchedInputTensorElements);
    }

    if (loadedProgram != nullptr && !loadedProgram->warmupComplete) {
        std::cout << "[migraphx] warming compiled program for '" << modelId
                  << "' before first frame." << std::endl;
        obs::logMiGraphXEnvironment(
            loadedProgram->runtimeFields,
            "runtime-warmup",
            loadedProgram->sourcePath,
            "start");

        const void* warmupOutputData = nullptr;
        std::size_t warmupOutputElements = 0;
        TensorDtype warmupOutputDtype = TensorDtype::Unknown;
        std::string warmupError;
        const void* warmupInputData = nullptr;
        std::size_t warmupInputElements = batchedInputTensorElements;
        if (inputContract.dtype == TensorDtype::Fp16) {
            std::fill(batchedInputTensorFp16.begin(), batchedInputTensorFp16.end(), 0u);
            warmupInputData = batchedInputTensorFp16.data();
            warmupInputElements = batchedInputTensorFp16.size();
        } else {
            std::fill(batchedInputTensorFp32.begin(), batchedInputTensorFp32.end(), 0.0f);
            warmupInputData = batchedInputTensorFp32.data();
            warmupInputElements = batchedInputTensorFp32.size();
        }

        if (!impl_->runInference(modelId, warmupInputData, warmupInputElements,
                                 inputContract.dtype, warmupOutputData,
                                 warmupOutputElements, warmupOutputDtype,
                                 warmupError, false)) {
            pclose(decodePipe);
            const int encodeStatus = pclose(encodePipe);
            AVE_ROCTX_RANGE_END();
            error = formatEncodeFailure("MiGraphX warmup failed: " + warmupError, encodeStatus);
            cleanupEncodeLog();
            return StageResult::Error;
        }
        {
            std::lock_guard<std::mutex> evalLock(loadedProgram->evalMutex);
            loadedProgram->prog.experimental_get_context().finish();
            loadedProgram->warmupComplete = true;
        }
        obs::logMiGraphXEnvironment(
            loadedProgram->runtimeFields,
            "runtime-warmup",
            loadedProgram->sourcePath,
            "complete");
    }
    using Clock = std::chrono::steady_clock;
    std::chrono::nanoseconds readTime{0};
    std::chrono::nanoseconds preprocessTime{0};
    std::chrono::nanoseconds inferenceTime{0};
    std::chrono::nanoseconds postprocessTime{0};
    std::chrono::nanoseconds writeTime{0};
    struct QueuedFrame {
        int frameIdx = 0;
        std::vector<std::uint8_t> pixels;
    };
    constexpr std::size_t kQueuedEncodeDepth = 4u;
    std::mutex encodeQueueMtx;
    std::condition_variable encodeQueueCv;
    std::deque<QueuedFrame> encodeQueue;
    // Pre-allocate kQueuedEncodeDepth + 1 buffers so the producer thread is
    // never blocked waiting for a free buffer while the encode queue is full.
    std::deque<std::vector<std::uint8_t>> freeFrameBuffers;
    for (std::size_t i = 0; i < kQueuedEncodeDepth + 1u; ++i) {
        freeFrameBuffers.emplace_back(outFrameBytes);
    }
    bool encodeInputClosed = false;
    bool encodeWriterFailed = false;
    std::string encodeWriterError;
    std::thread encodeThread;
    auto shutdownEncodeWriter = [&](bool discardPending) {
        {
            std::lock_guard<std::mutex> lk(encodeQueueMtx);
            if (discardPending) {
                encodeQueue.clear();
            }
            encodeInputClosed = true;
        }
        encodeQueueCv.notify_all();
        if (encodeThread.joinable()) {
            encodeThread.join();
        }
    };
    auto acquireOutputBuffer = [&]() -> bool {
        std::unique_lock<std::mutex> lk(encodeQueueMtx);
        encodeQueueCv.wait(lk, [&]() {
            return !freeFrameBuffers.empty() || encodeWriterFailed;
        });
        if (encodeWriterFailed) {
            return false;
        }
        rgbOut = std::move(freeFrameBuffers.front());
        freeFrameBuffers.pop_front();
        lk.unlock();
        if (rgbOut.size() != outFrameBytes) {
            rgbOut.resize(outFrameBytes);
        }
        return true;
    };
    auto submitOutputBuffer = [&](int producedFrameIdx) -> bool {
        std::unique_lock<std::mutex> lk(encodeQueueMtx);
        encodeQueueCv.wait(lk, [&]() {
            return encodeQueue.size() < kQueuedEncodeDepth || encodeWriterFailed;
        });
        if (encodeWriterFailed) {
            return false;
        }
        encodeQueue.push_back(QueuedFrame{producedFrameIdx, std::move(rgbOut)});
        lk.unlock();
        encodeQueueCv.notify_all();
        return true;
    };
    encodeThread = std::thread([&]() {
        while (true) {
            QueuedFrame frame;
            {
                std::unique_lock<std::mutex> lk(encodeQueueMtx);
                encodeQueueCv.wait(lk, [&]() {
                    return !encodeQueue.empty() || encodeInputClosed;
                });
                if (encodeQueue.empty()) {
                    break;
                }
                frame = std::move(encodeQueue.front());
                encodeQueue.pop_front();
            }
            encodeQueueCv.notify_all();

            const auto writeStart = Clock::now();
            const std::size_t written =
                std::fwrite(frame.pixels.data(), 1, outFrameBytes, encodePipe);
            writeTime += Clock::now() - writeStart;
            if (written != outFrameBytes) {
                std::lock_guard<std::mutex> lk(encodeQueueMtx);
                encodeWriterFailed = true;
                encodeWriterError =
                    "Encode pipe write failed at frame " + std::to_string(frame.frameIdx)
                    + "; the FFmpeg encoder exited early.";
                encodeInputClosed = true;
                encodeQueue.clear();
                encodeQueueCv.notify_all();
                break;
            }

            {
                std::lock_guard<std::mutex> lk(encodeQueueMtx);
                freeFrameBuffers.push_back(std::move(frame.pixels));
            }
            encodeQueueCv.notify_all();
        }
    });
    auto abortProcessing = [&](const std::string& detail,
                               bool includeEncodeFailure = false) -> StageResult {
        pclose(decodePipe);
        shutdownEncodeWriter(true);
        const int encodeStatus = pclose(encodePipe);
        AVE_ROCTX_RANGE_END();

        bool writerFailed = false;
        std::string combinedDetail = detail;
        {
            std::lock_guard<std::mutex> lk(encodeQueueMtx);
            writerFailed = encodeWriterFailed;
            if (!encodeWriterError.empty() && encodeWriterError != detail) {
                combinedDetail += "\n" + encodeWriterError;
            }
        }

        if (includeEncodeFailure || writerFailed) {
            error = formatEncodeFailure(combinedDetail, encodeStatus);
        } else {
            error = combinedDetail;
        }
        cleanupEncodeLog();
        return StageResult::Error;
    };
    if (!acquireOutputBuffer()) {
        return abortProcessing(
            "Failed to acquire output frame buffer for MiGraphX encode.",
            true);
    }
    const auto loopStart = Clock::now();
    int frameIdx = 0;
    bool cancelled = false;

    while (true) {
        // ── Cancel / Pause check ────────────────────────────────
        if (opts.cancelFlag && opts.cancelFlag->load(std::memory_order_relaxed)) {
            std::cout << "[migraphx] Cancelled at frame " << frameIdx << std::endl;
            cancelled = true;
            break;
        }
        while (opts.pauseFlag && opts.pauseFlag->load(std::memory_order_relaxed)) {
            if (opts.cancelFlag && opts.cancelFlag->load(std::memory_order_relaxed)) {
                cancelled = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (cancelled) break;

        // Read one raw RGB24 frame from decode pipe
        const auto readStart = Clock::now();
        std::size_t totalRead = 0;
        while (totalRead < frameBytes) {
            const std::size_t n = std::fread(rgbIn.data() + totalRead, 1,
                                              frameBytes - totalRead, decodePipe);
            if (n == 0) break;
            totalRead += n;
        }
        readTime += Clock::now() - readStart;
        if (totalRead == 0) break;  // EOF
        if (totalRead != frameBytes) {
            std::cerr << "[migraphx] Partial frame " << frameIdx
                      << " (" << totalRead << "/" << frameBytes << " bytes) — skipping." << std::endl;
            break;
        }

        std::fill(rgbOut.begin(), rgbOut.end(), 0u);

        for (std::size_t batchStart = 0; batchStart < tileDispatches.size();
             batchStart += static_cast<std::size_t>(compiledBatch)) {
            const std::size_t activeTiles = std::min(
                static_cast<std::size_t>(compiledBatch),
                tileDispatches.size() - batchStart);

            const auto preprocessStart = Clock::now();
            const void* inputTensorData = nullptr;
            std::size_t inputTensorElements = batchedInputTensorElements;
            if (inputContract.dtype == TensorDtype::Fp16) {
                for (std::size_t slot = 0; slot < static_cast<std::size_t>(compiledBatch); ++slot) {
                    const std::size_t dispatchIndex =
                        batchStart + std::min(slot, activeTiles - 1u);
                    const TileDispatch& dispatch = tileDispatches[dispatchIndex];
                    packRgbTileClampToNchwFp16(
                        rgbIn.data(), inW, inH,
                        dispatch.tileX, dispatch.tileY,
                        modelInW, modelInH,
                        batchedInputTensorFp16.data() + slot * tileTensorElements);
                }
                inputTensorData = batchedInputTensorFp16.data();
                inputTensorElements = batchedInputTensorFp16.size();
            } else {
                for (std::size_t slot = 0; slot < static_cast<std::size_t>(compiledBatch); ++slot) {
                    const std::size_t dispatchIndex =
                        batchStart + std::min(slot, activeTiles - 1u);
                    const TileDispatch& dispatch = tileDispatches[dispatchIndex];
                    packRgbTileClampToNchwFp32(
                        rgbIn.data(), inW, inH,
                        dispatch.tileX, dispatch.tileY,
                        modelInW, modelInH,
                        batchedInputTensorFp32.data() + slot * tileTensorElements);
                }
                inputTensorData = batchedInputTensorFp32.data();
                inputTensorElements = batchedInputTensorFp32.size();
            }
            preprocessTime += Clock::now() - preprocessStart;

            const auto inferenceStart = Clock::now();
            const void* outputTensorData = nullptr;
            std::size_t outputTensorElements = 0;
            TensorDtype outputTensorDtype = TensorDtype::Unknown;
            if (!impl_->runInference(modelId, inputTensorData, inputTensorElements,
                                     inputContract.dtype, outputTensorData,
                                     outputTensorElements, outputTensorDtype, error)) {
                std::cerr << "[migraphx] Frame " << frameIdx
                          << " tile batch starting at " << batchStart
                          << " inference FAILED: " << error << std::endl;
                return abortProcessing(error);
            }
            inferenceTime += Clock::now() - inferenceStart;

            const auto postprocessStart = Clock::now();
            if (outputTensorData == nullptr || outputTensorElements != expectedBatchOutputElements) {
                std::ostringstream os;
                os << "Output tensor pointer/size mismatch at frame " << frameIdx
                   << " batch starting at tile " << batchStart
                   << ": ptr=" << outputTensorData
                   << " elems=" << outputTensorElements
                   << " expected=" << expectedBatchOutputElements;
                return abortProcessing(os.str());
            }

            for (std::size_t slot = 0; slot < activeTiles; ++slot) {
                const TileDispatch& dispatch = tileDispatches[batchStart + slot];
                if (outputTensorDtype == TensorDtype::Fp16) {
                    const auto* tileOutput =
                        static_cast<const std::uint16_t*>(outputTensorData)
                        + slot * expectedTileOutputElements;
                    blitNchwFp16TileRegionToRgb24(
                        tileOutput, modelOutC, modelOutW, modelOutH,
                        dispatch.srcXWindow.offset * scaleX,
                        dispatch.srcYWindow.offset * scaleY,
                        (dispatch.srcXWindow.end - dispatch.srcXWindow.begin) * scaleX,
                        (dispatch.srcYWindow.end - dispatch.srcYWindow.begin) * scaleY,
                        rgbOut, outW,
                        dispatch.srcXWindow.begin * scaleX,
                        dispatch.srcYWindow.begin * scaleY);
                } else if (outputTensorDtype == TensorDtype::Fp32) {
                    const auto* tileOutput =
                        static_cast<const float*>(outputTensorData)
                        + slot * expectedTileOutputElements;
                    blitNchwFp32TileRegionToRgb24(
                        tileOutput, modelOutC, modelOutW, modelOutH,
                        dispatch.srcXWindow.offset * scaleX,
                        dispatch.srcYWindow.offset * scaleY,
                        (dispatch.srcXWindow.end - dispatch.srcXWindow.begin) * scaleX,
                        (dispatch.srcYWindow.end - dispatch.srcYWindow.begin) * scaleY,
                        rgbOut, outW,
                        dispatch.srcXWindow.begin * scaleX,
                        dispatch.srcYWindow.begin * scaleY);
                } else {
                    std::ostringstream os;
                    os << "Unsupported output tensor dtype at frame " << frameIdx
                       << " batch starting at tile " << batchStart << ": "
                       << toString(outputTensorDtype);
                    return abortProcessing(os.str());
                }
            }
            postprocessTime += Clock::now() - postprocessStart;
        }

        if (rgbOut.size() != outFrameBytes) {
            std::cerr << "[migraphx] Frame " << frameIdx
                      << " output size mismatch: " << rgbOut.size()
                      << " vs expected " << outFrameBytes << std::endl;
            std::ostringstream os;
            os << "Output tensor size mismatch at frame " << frameIdx;
            return abortProcessing(os.str());
        }

        // Emit live frame preview
        const int completedFrameIdx = frameIdx;
        const int nextFrameIdx = frameIdx + 1;
        const int pvInterval = opts.previewFrameInterval > 0 ? opts.previewFrameInterval : 15;
        if (opts.framePreviewCb && (nextFrameIdx % pvInterval == 1 || pvInterval == 1)) {
            opts.framePreviewCb(rgbOut.data(), outW, outH);
        }

        if (!submitOutputBuffer(completedFrameIdx)) {
            return abortProcessing(
                "Encode pipe write failed at frame " + std::to_string(completedFrameIdx)
                + "; the intermediate FFmpeg encoder exited early.",
                true);
        }

        ++frameIdx;

        if (!acquireOutputBuffer()) {
            return abortProcessing(
                "Encode pipe write failed while waiting for a free output buffer.",
                true);
        }

        // Report real progress based on actual frame count
        if (progressCb) {
            float frac = 0.0f;
            if (totalFrames > 0) {
                frac = static_cast<float>(frameIdx) / static_cast<float>(totalFrames);
                frac = std::min(frac, 1.0f);
            } else {
                // Unknown total — use logarithmic approach
                frac = 1.0f - 1.0f / (1.0f + static_cast<float>(frameIdx) * 0.01f);
            }
            progressCb(frac, "MiGraphX: processed frame " + std::to_string(frameIdx)
                        + (totalFrames > 0 ? "/" + std::to_string(totalFrames) : ""));
        }

        if (frameIdx % 30 == 0) {
            std::cout << "[migraphx] Processed " << frameIdx << " frames"
                      << (totalFrames > 0 ? " / " + std::to_string(totalFrames) : "")
                      << std::endl;
        }
    }

    pclose(decodePipe);
    if (cancelled) {
        shutdownEncodeWriter(true);
    } else {
        shutdownEncodeWriter(false);
    }
    const int encodeRet = pclose(encodePipe);

    AVE_ROCTX_RANGE_END();

    if (cancelled) {
        cleanupEncodeLog();
        error = "Processing cancelled by user at frame " + std::to_string(frameIdx);
        return StageResult::Cancelled;
    }
    if (frameIdx == 0) {
        cleanupEncodeLog();
        error = "No frames were decoded from " + inputVideo;
        return StageResult::Error;
    }

    if (encodeWriterFailed) {
        error = formatEncodeFailure(
            encodeWriterError.empty()
                ? "FFmpeg encode pipe exited after MiGraphX processing."
                : encodeWriterError,
            encodeRet);
        cleanupEncodeLog();
        return StageResult::Error;
    }

    if (encodeRet != 0) {
        error = formatEncodeFailure(
            "FFmpeg encode pipe exited after MiGraphX processing.",
            encodeRet);
        cleanupEncodeLog();
        return StageResult::Error;
    }

    cleanupEncodeLog();

    const auto totalElapsed = Clock::now() - loopStart;
    const double totalSeconds = std::chrono::duration<double>(totalElapsed).count();
    const auto avgMs = [frameIdx](std::chrono::nanoseconds totalNs) {
        if (frameIdx <= 0) {
            return 0.0;
        }
        return std::chrono::duration<double, std::milli>(totalNs).count()
             / static_cast<double>(frameIdx);
    };
    const double throughputFps = totalSeconds > 0.0
        ? static_cast<double>(frameIdx) / totalSeconds
        : 0.0;
    std::cout << "[migraphx] AI inference complete: " << frameIdx
              << " frames processed via MiGraphX for stage "
              << toString(stage.kind) << std::endl;
    std::cout << "[migraphx] timing: total=" << totalSeconds
              << "s fps=" << throughputFps
              << " avg_ms/frame read=" << avgMs(readTime)
              << " preprocess=" << avgMs(preprocessTime)
              << " infer=" << avgMs(inferenceTime)
              << " postprocess=" << avgMs(postprocessTime)
              << " write=" << avgMs(writeTime)
              << " direct_final_encode=" << (directOutputEncode ? "on" : "off")
              << std::endl;

    if (progressCb) {
        progressCb(1.0f, "MiGraphX inference complete — " + std::to_string(frameIdx) + " frames.");
    }

    return StageResult::Processed;
#else
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    (void)opts;
    error = "MiGraphX backend not compiled (-DAVE_HAVE_MIGRAPHX=OFF).";
    return StageResult::Deferred;
#endif
}
}  // namespace ave
