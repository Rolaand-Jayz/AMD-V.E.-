// ─────────────────────────────────────────────────────────────────
// error_taxonomy.cpp — InferenceError implementation
// ─────────────────────────────────────────────────────────────────
#include "ave/error_taxonomy.hpp"

namespace ave {

std::string toString(InferenceErrorKind kind) {
    switch (kind) {
        case InferenceErrorKind::ModelIncompatible:   return "ModelIncompatible";
        case InferenceErrorKind::CompileFailure:      return "CompileFailure";
        case InferenceErrorKind::ArtifactInvalid:     return "ArtifactInvalid";
        case InferenceErrorKind::VulkanDeviceFailure: return "VulkanDeviceFailure";
        case InferenceErrorKind::InteropFailure:      return "InteropFailure";
        case InferenceErrorKind::SyncHazard:          return "SyncHazard";
        case InferenceErrorKind::FFmpegInteropFailure:return "FFmpegInteropFailure";
        case InferenceErrorKind::RuntimeFailure:      return "RuntimeFailure";
    }
    return "Unknown";
}

std::string InferenceError::format() const {
    std::string out = "[" + toString(kind) + "] " + message;
    if (!context.empty()) { out += "\n  context: " + context; }
    return out;
}

InferenceError InferenceError::modelIncompatible(const std::string& msg,
                                                  const std::string& ctx) {
    return {InferenceErrorKind::ModelIncompatible, msg, ctx};
}
InferenceError InferenceError::compileFailure(const std::string& msg,
                                               const std::string& ctx) {
    return {InferenceErrorKind::CompileFailure, msg, ctx};
}
InferenceError InferenceError::artifactInvalid(const std::string& msg,
                                                const std::string& ctx) {
    return {InferenceErrorKind::ArtifactInvalid, msg, ctx};
}
InferenceError InferenceError::vulkanDevice(const std::string& msg,
                                             const std::string& ctx) {
    return {InferenceErrorKind::VulkanDeviceFailure, msg, ctx};
}
InferenceError InferenceError::interopFailure(const std::string& msg,
                                               const std::string& ctx) {
    return {InferenceErrorKind::InteropFailure, msg, ctx};
}
InferenceError InferenceError::syncHazard(const std::string& msg,
                                           const std::string& ctx) {
    return {InferenceErrorKind::SyncHazard, msg, ctx};
}
InferenceError InferenceError::ffmpegInterop(const std::string& msg,
                                              const std::string& ctx) {
    return {InferenceErrorKind::FFmpegInteropFailure, msg, ctx};
}
InferenceError InferenceError::runtimeFailure(const std::string& msg,
                                               const std::string& ctx) {
    return {InferenceErrorKind::RuntimeFailure, msg, ctx};
}

}  // namespace ave
