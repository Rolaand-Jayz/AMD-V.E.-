// ─────────────────────────────────────────────────────────────────
// GLSL Shader Backend – full implementation
// Applies GLSL fragment/compute shaders to video frames via
// FFmpeg's libplacebo filter or mpv's GPU shader pipeline.
//
// Conditionally compiled against AVE_HAVE_GLSL.
// When absent, the backend can still work if ffmpeg is built with
// libplacebo support (detected at runtime).
// ─────────────────────────────────────────────────────────────────
#include "ave/backends/glsl_shader_backend.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ave/frame_io.hpp"
#include "ave/filter_catalog.hpp"
#include "ave/stage.hpp"
#include "ave/types.hpp"

namespace ave {
namespace {

bool fileExistsGL(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

bool commandInPathGL(const std::string& cmd) {
    const char* pathEnv = std::getenv("PATH");
    if (!pathEnv) { return false; }
    std::string path(pathEnv);
    std::size_t start = 0;
    while (start <= path.size()) {
        std::size_t end = path.find(':', start);
        if (end == std::string::npos) { end = path.size(); }
        const std::string dir = path.substr(start, end - start);
        if (!dir.empty() && fileExistsGL((std::filesystem::path(dir) / cmd).string())) {
            return true;
        }
        if (end == path.size()) { break; }
        start = end + 1;
    }
    return false;
}

std::string quoteArgGL(const std::string& value) {
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

// Check if ffmpeg was built with libplacebo support.
bool ffmpegHasLibplacebo() {
    const int rc = std::system(
        "ffmpeg -hide_banner -filters 2>/dev/null | grep -q libplacebo");
    return rc == 0;
}

// Check if mpv is available with GPU shader support.
bool mpvAvailable() {
    return commandInPathGL("mpv");
}

// Base directory for shader files.
std::string shaderBaseDir() {
    const char* home = std::getenv("HOME");
    if (!home) { return "/usr/share/ave/shaders"; }
    return std::string(home) + "/.local/share/ave/shaders";
}

// Map from stage kind to recommended default shader filenames.
// Users can override via the "shader" stage parameter.
const std::unordered_map<StageKind, std::vector<std::string>>& defaultShaderMap() {
    static const std::unordered_map<StageKind, std::vector<std::string>> map = {
        {StageKind::Upscale,    {"FSRCNNX_x2_16-0-4-1.glsl",
                                  "SSimSuperRes.glsl",
                                  "KrigBilateral.glsl"}},
        {StageKind::Sharpen,    {"adaptive-sharpen.glsl"}},
        {StageKind::Denoise,    {"nlmeans.glsl",
                                  "nnedi3-nns64-win8x4.glsl"}},
        {StageKind::Dehalo,     {"dehalo.glsl"}},
        {StageKind::ColorFix,   {"colorlevels.glsl"}},
    };
    return map;
}

// Resolve shader paths for a stage: user-specified or defaults.
std::vector<std::string> resolveShaders(const EnhancementStage& stage) {
    // Check if user specified explicit shader path(s).
    auto it = stage.params.find("shader");
    if (it != stage.params.end()) {
        if (const auto* s = std::get_if<std::string>(&it->second)) {
            if (!s->empty()) {
                // Support comma-separated shader paths.
                std::vector<std::string> paths;
                std::istringstream iss(*s);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    // Trim whitespace
                    std::size_t b = 0;
                    while (b < token.size() && std::isspace(static_cast<unsigned char>(token[b]))) { ++b; }
                    std::size_t e = token.size();
                    while (e > b && std::isspace(static_cast<unsigned char>(token[e - 1]))) { --e; }
                    if (b < e) { paths.push_back(token.substr(b, e - b)); }
                }
                return paths;
            }
        }
    }

    // Use defaults for this stage kind.
    const auto& map = defaultShaderMap();
    auto defIt = map.find(stage.kind);
    if (defIt == map.end()) { return {}; }

    // Resolve relative names against the shader base directory.
    const std::string base = shaderBaseDir();
    std::vector<std::string> resolved;
    for (const auto& namev : defIt->second) {
        const std::string full = base + "/" + namev;
        if (fileExistsGL(full)) {
            resolved.push_back(full);
        }
    }
    return resolved;
}

// Write catalog-embedded GLSL shaders to temporary files and
// return their paths for inclusion in the shader chain.
std::vector<std::string> writeCatalogShaders(
        const std::vector<ActiveFilter>& catalogFilters,
        StageKind stageKind) {
    std::vector<std::string> result;
    const std::string tmpDir = shaderBaseDir() + "/tmp_catalog";
    std::error_code ec;
    std::filesystem::create_directories(tmpDir, ec);

    for (const auto& af : catalogFilters) {
        if (!af.enabled) { continue; }
        const auto* ef = findFilter(af.id);
        if (!ef) { continue; }
        if (ef->runtime != FilterRuntime::Glsl) { continue; }
        if (ef->stageKind != stageKind) { continue; }

        // Resolve parameter substitutions.
        std::string src = resolveSource(*ef, af.paramValues);

        // Write to a temp file.
        const std::string path = tmpDir + "/" + af.id + ".glsl";
        std::ofstream ofs(path);
        if (ofs) {
            ofs << src;
            ofs.close();
            result.push_back(path);
            std::cout << "[glsl-catalog] Wrote embedded shader: " << path << std::endl;
        }
    }
    return result;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────
// Static helper
// ─────────────────────────────────────────────────────────────────
std::vector<std::string> GlslShaderBackend::defaultShadersForStage(StageKind kind) {
    const auto& map = defaultShaderMap();
    auto it = map.find(kind);
    if (it == map.end()) { return {}; }
    return it->second;
}

// ─────────────────────────────────────────────────────────────────
// Impl
// ─────────────────────────────────────────────────────────────────

struct GlslShaderBackend::Impl {
    bool initialised     = false;
    bool hasLibplacebo   = false;
    bool hasMpv          = false;
    std::vector<ActiveFilter> catalogFilters;
};

GlslShaderBackend::GlslShaderBackend() : impl_(std::make_unique<Impl>()) {}
GlslShaderBackend::~GlslShaderBackend() = default;

BackendType GlslShaderBackend::type() const { return BackendType::GlslShader; }
std::string GlslShaderBackend::name() const { return "GLSL Shader"; }
bool GlslShaderBackend::supportsDirectOutputEncode() const {
    return impl_->initialised ? impl_->hasLibplacebo : ffmpegHasLibplacebo();
}

bool GlslShaderBackend::isAvailable(std::string& reason) const {
    const bool placebo = ffmpegHasLibplacebo();
    const bool mpv     = mpvAvailable();

    if (!placebo && !mpv) {
        reason = "Neither FFmpeg libplacebo filter nor mpv found. "
                 "Install ffmpeg with libplacebo support or install mpv.";
        return false;
    }

    std::ostringstream os;
    os << "GLSL shader processing available via";
    if (placebo) { os << " libplacebo"; }
    if (placebo && mpv) { os << " and"; }
    if (mpv) { os << " mpv"; }
    os << ".";
    reason = os.str();
    return true;
}

bool GlslShaderBackend::initialize(std::string& error) {
    if (impl_->initialised) { return true; }

    impl_->hasLibplacebo = ffmpegHasLibplacebo();
    impl_->hasMpv        = mpvAvailable();

    if (!impl_->hasLibplacebo && !impl_->hasMpv) {
        error = "No GLSL shader runtime available.";
        return false;
    }

    // Ensure shader directory exists.
    const std::string baseDir = shaderBaseDir();
    std::error_code ec;
    std::filesystem::create_directories(baseDir, ec);
    if (ec) {
        std::cerr << "[glsl] Warning: could not create shader directory: "
                  << baseDir << std::endl;
    }

    impl_->initialised = true;
    std::cout << "[glsl] Backend initialised."
              << " libplacebo=" << (impl_->hasLibplacebo ? "yes" : "no")
              << " mpv=" << (impl_->hasMpv ? "yes" : "no")
              << std::endl;
    return true;
}

StageResult GlslShaderBackend::runStage(const EnhancementStage& /*stage*/,
                                         std::string& /*error*/) {
    // Pre-validation only; actual work is in processVideoFile.
    return StageResult::Deferred;
}

StageResult GlslShaderBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
    if (!impl_->initialised) {
        if (!initialize(error)) {
            return StageResult::Deferred;
        }
    }

    // Resolve shaders: user-specified paths + catalog-embedded shaders.
    std::vector<std::string> shaderPaths = resolveShaders(stage);
    auto catalogPaths = writeCatalogShaders(impl_->catalogFilters, stage.kind);
    shaderPaths.insert(shaderPaths.end(), catalogPaths.begin(), catalogPaths.end());

    if (shaderPaths.empty()) {
        // No shaders available for this stage — defer to FFmpeg filters.
        return StageResult::Deferred;
    }

    // Determine target output dimensions (if upscaling).
    int outW = 0;
    int outH = 0;
    {
        auto wit = stage.params.find("width");
        auto hit = stage.params.find("height");
        if (wit != stage.params.end()) {
            if (const auto* d = std::get_if<std::int64_t>(&wit->second))
                outW = static_cast<int>(*d);
        }
        if (hit != stage.params.end()) {
            if (const auto* d = std::get_if<std::int64_t>(&hit->second))
                outH = static_cast<int>(*d);
        }
    }

    if (progressCb) { progressCb(0.0F, "Applying GLSL shaders..."); }

    if (impl_->hasLibplacebo) {
        const bool directOutputEncode = opts.directOutputEncode;

        // ── Direct FFmpeg libplacebo pipeline ────────────────────
        std::ostringstream cmd;
        cmd << "ffmpeg -hide_banner -loglevel error -y ";
        if (opts.previewDurationSec > 0.0) {
            cmd << "-t " << opts.previewDurationSec << " ";
        }
        cmd << "-i " << quoteArgGL(inputVideo) << ' ';
        if (directOutputEncode) {
            cmd << "-i " << quoteArgGL(inputVideo) << ' ';
        }
        cmd << "-vf \"";
        for (std::size_t i = 0; i < shaderPaths.size(); ++i) {
            if (i > 0) { cmd << ","; }
            cmd << "libplacebo=custom_shader_path='" << shaderPaths[i] << "'";
            if (outW > 0 && outH > 0 && i == shaderPaths.size() - 1) {
                cmd << ":w=" << outW << ":h=" << outH;
            }
        }
        if (directOutputEncode) {
            cmd << ",format=yuv420p";
        }
        cmd << "\" ";

        if (directOutputEncode) {
            cmd << "-map 0:v:0 -map 1:a? "
                << "-c:v " << opts.outputCodec << ' '
                << "-crf " << opts.outputCrf << ' '
                << "-preset " << opts.outputPreset << ' ';
            if (!opts.outputProfile.empty()) {
                cmd << "-profile:v " << opts.outputProfile << ' ';
            }
            if (opts.outputThreads > 0) {
                cmd << "-threads " << opts.outputThreads << ' ';
            }
            cmd << "-c:a copy -shortest ";
        } else {
            cmd << "-map 0:v:0 -map 0:a? "
                << "-c:v libx264 -crf 0 -preset ultrafast -c:a copy ";
        }

        cmd << quoteArgGL(outputVideo) << " 2>&1";

        std::cout << "[glsl] Running libplacebo pipeline: "
                  << shaderPaths.size() << " shaders"
                  << (directOutputEncode ? " with direct final encode" : "")
                  << std::endl;
        const int rc = std::system(cmd.str().c_str());
        if (rc != 0) {
            error = "FFmpeg libplacebo exited with code " + std::to_string(rc);
            return StageResult::Error;
        }
    } else if (impl_->hasMpv) {
        // ── mpv GPU shader pipeline ─────────────────────────────
        std::ostringstream cmd;
        cmd << "mpv --no-config --vo=gpu-next --gpu-api=vulkan "
            << "--no-audio --no-sub ";
        for (const auto& sp : shaderPaths) {
            cmd << "--glsl-shader=\"" << sp << "\" ";
        }
        if (outW > 0 && outH > 0) {
            cmd << "--vf=scale=" << outW << ":" << outH << " ";
        }
        cmd << "--o=\"" << outputVideo << "\" "
            << "--of=mp4 --ovc=libx264 --ovcopts=crf=0,preset=ultrafast ";
        if (opts.previewDurationSec > 0.0) {
            cmd << "--end=" << opts.previewDurationSec << " ";
        }
        cmd << "\"" << inputVideo << "\" 2>&1";

        std::cout << "[glsl] Running mpv pipeline: "
                  << shaderPaths.size() << " shaders" << std::endl;
        const int rc = std::system(cmd.str().c_str());
        if (rc != 0) {
            error = "mpv shader processing exited with code " + std::to_string(rc);
            return StageResult::Error;
        }
    } else {
        error = "No GLSL runtime available (need ffmpeg with libplacebo or mpv).";
        return StageResult::Deferred;
    }

    if (progressCb) { progressCb(1.0F, "GLSL shader processing complete."); }
    std::cout << "[glsl] Done: " << toString(stage.kind)
              << " with " << shaderPaths.size() << " shaders" << std::endl;
    return StageResult::Processed;
}

void GlslShaderBackend::setCatalogFilters(
        const std::vector<ActiveFilter>& filters) {
    impl_->catalogFilters = filters;
}

}  // namespace ave
