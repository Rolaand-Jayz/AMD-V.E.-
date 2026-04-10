#include "ave/model_catalog.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

namespace ave {

namespace {

struct BuiltinFamilyInfo {
    const char* id;
    const char* name;
    bool supportsFusedExecution;
    bool supportsSelectiveCapabilities;
};

const BuiltinFamilyInfo* builtinFamilyInfoForId(const std::string& modelId) {
    static const std::unordered_map<std::string, BuiltinFamilyInfo> map = {
        {"nomos2-otf-x4", {"nomos2-otf", "Nomos2 OTF ESRGAN", true, false}},
        {"realesrgan-x4-restore", {"realesrgan-x4", "Real-ESRGAN x4", true, false}},
        {"realesrgan-x4-general", {"realesrgan-x4", "Real-ESRGAN x4", true, false}},
        {"realesrgan-x4plus-pth", {"realesrgan-x4plus-pth", "Real-ESRGAN x4+ (PyTorch)", false, false}},
        {"realesrgan-x4-axera", {"realesrgan-x4-axera", "Real-ESRGAN x4 (AXERA export)", false, false}},
        {"openproteus-compact-x2", {"openproteus-compact-x2", "OpenProteus Compact x2", true, false}},
        {"realesrgan-x4plus-ncnn", {"realesrgan-x4plus-ncnn", "Real-ESRGAN x4+ NCNN", false, false}},
        {"realesrgan-x4plus-anime-ncnn", {"realesrgan-x4plus-anime-ncnn", "Real-ESRGAN x4+ Anime NCNN", false, false}},
        {"realesr-animevideov3-x2-ncnn", {"realesr-animevideov3-x2-ncnn", "Real-ESRGAN AnimeVideo v3 x2 NCNN", false, false}},
        {"realesr-animevideov3-x3-ncnn", {"realesr-animevideov3-x3-ncnn", "Real-ESRGAN AnimeVideo v3 x3 NCNN", false, false}},
        {"realesr-animevideov3-x4-ncnn", {"realesr-animevideov3-x4-ncnn", "Real-ESRGAN AnimeVideo v3 x4 NCNN", false, false}},
        {"realesrnet-x2plus", {"realesrnet-x2plus", "Real-ESRNet x2+", true, false}},
        {"realesrnet-deblur-x2", {"realesrnet-x2plus", "Real-ESRNet x2+", true, false}},
        {"nmkd-siax-x4", {"nmkd-siax-x4", "NMKD Siax 200k", true, false}},
        {"ultrasharp-x4", {"ultrasharp-x4", "UltraSharp x4", true, false}},
        {"wtp-uds-esrgan-x4", {"wtp-uds-esrgan-x4", "WTP-UDS-ESRGAN x4", true, false}},
        {"nafnet-denoise", {"nafnet-restoration", "NAFNet Blind Restoration", true, false}},
        {"nafnet-dehalo", {"nafnet-restoration", "NAFNet Blind Restoration", true, false}},
        {"openproteus-deblur-x2", {"openproteus-compact-x2", "OpenProteus Compact x2", true, false}},
        {"clearreality-x4-denoise", {"clearreality-v1", "ClearReality V1", true, false}},
        {"clearreality-deblur-x4", {"clearreality-v1", "ClearReality V1", true, false}},
        {"clearreality-x4-fast", {"clearreality-v1", "ClearReality V1", true, false}},
        {"remacri-x4", {"remacri-x4", "Remacri x4", true, false}},
        {"ultrasharpv2-x4", {"ultrasharpv2", "UltraSharp V2", true, false}},
        {"ultrasharpv2-lite-x4", {"ultrasharpv2-lite", "UltraSharp V2 Lite", true, false}},
        {"iqa-color-enhance", {"parametric-color-fix", "Parametric Color Fix", false, false}},
        {"swinir-color", {"swinir-x4", "SwinIR x4", true, false}},
        {"swinir-x4-general", {"swinir-x4", "SwinIR x4", true, false}},
        {"animesharp-x4", {"animesharp-x4", "AnimeSharp x4", true, false}},
        {"modernspanimation-x2", {"modernspanimation-v2", "ModernSpanimation v2 x2", true, false}},
        {"modernspanimation-x2-fp32", {"modernspanimation-v2-fp32", "ModernSpanimation v2 x2 fp32", true, false}},
        {"modernspanimation-x2-v1compact", {"modernspanimation-v1-compact", "ModernSpanimation v1 Compact x2", true, false}},
        {"animejananai-hd-compact-x2", {"animejanai-hd-v3", "AnimeJaNai HD V3 Compact x2", true, false}},
        {"realistic-rescaler-x4", {"realistic-rescaler-x4", "RealisticRescaler x4", true, false}},
        {"sharpen-cas", {"cas-sharpen", "Contrast-Adaptive Sharpening", false, false}},
        {"rife-v4-7", {"rife-v4-7", "RIFE v4.7", false, false}},
        {"rife-v4-8", {"rife-v4-8", "RIFE v4.8", false, false}},
        {"rife-v4-9", {"rife-v4-9", "RIFE v4.9", false, false}},
        {"interp-ffmpeg", {"interp-ffmpeg", "FFmpeg Interpolation", false, false}},
    };
    const auto it = map.find(modelId);
    if (it == map.end()) {
        return nullptr;
    }
    return &it->second;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────
// Built-in model catalog
// ─────────────────────────────────────────────────────────────────
// Covers the same model families supported by the upstream
// reference application (compression restore, artifact removal,
// denoise, deblur, dehalo, color fix, upscale, sharpen,
// interpolate) while relying solely on AMD ROCm / MiGraphX and
// NCNN-Vulkan runtimes.
// ─────────────────────────────────────────────────────────────────

static const std::vector<ModelEntry> kCatalog = {

    // ════════════════════════════════════════════════════════════
    // Compression Restore / DeBlock / DeH264
    // ════════════════════════════════════════════════════════════
    {
        /* id          */ "nomos2-otf-x4",
        /* displayName */ "4xNomos2 OTF ESRGAN (compression artifact removal)",
        /* stage       */ StageKind::RestoreCompression,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4xNomos2_otf_esrgan.onnx",
        /* filename    */ "4xNomos2_otf_esrgan.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "ESRGAN trained with on-the-fly (OTF) JPEG/video compression degradations; simultaneous artifact removal and x4 upscale.",
        /* isDefault   */ true,
        /* minVram     */ 2048
    },
    {
        /* id          */ "realesrgan-x4-restore",
        /* displayName */ "Real-ESRGAN x4 (restoration + upscale)",
        /* stage       */ StageKind::RestoreCompression,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/RealESRGAN_x4.onnx",
        /* filename    */ "RealESRGAN_x4.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Real-ESRGAN x4plus general-purpose restoration and upscale; handles JPEG, H.264, and mixed degradations.",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },
    {
        /* id          */ "nmkd-siax-x4",
        /* displayName */ "NMKD Siax 200k (versatile restoration + upscale)",
        /* stage       */ StageKind::RestoreCompression,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x_NMKD-Siax_200k.onnx",
        /* filename    */ "4x_NMKD-Siax_200k.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "NMKD Siax 200k – versatile general enhancement model handling noise, blur, and compression at x4.",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },

    // ════════════════════════════════════════════════════════════
    // Artifact Removal (blocking, ringing, banding)
    // ════════════════════════════════════════════════════════════
    {
        /* id          */ "ultrasharp-x4",
        /* displayName */ "4x-UltraSharp (universal artifact removal + upscale)",
        /* stage       */ StageKind::RemoveArtifacts,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-UltraSharp.onnx",
        /* filename    */ "4x-UltraSharp.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "4x-UltraSharp ESRGAN – removes blocking, ringing, and banding while producing crisp x4 output.",
        /* isDefault   */ true,
        /* minVram     */ 2048
    },
    {
        /* id          */ "wtp-uds-esrgan-x4",
        /* displayName */ "4x WTP-UDS-ESRGAN (detail-preserving restoration)",
        /* stage       */ StageKind::RemoveArtifacts,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-WTP-UDS-Esrgan.onnx",
        /* filename    */ "4x-WTP-UDS-Esrgan.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "WTP-UDS ESRGAN x4 – excellent edge and texture fidelity for blocking and ringing artifact removal.",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },

    // ════════════════════════════════════════════════════════════
    // Denoise
    // ════════════════════════════════════════════════════════════
    {
        /* id          */ "nafnet-denoise",
        /* displayName */ "NAFNet Blind Restoration (denoising)",
        /* stage       */ StageKind::Denoise,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/opencv/deblurring_nafnet/resolve/main/deblurring_nafnet_2025may.onnx",
        /* filename    */ "deblurring_nafnet_2025may.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "NAFNet nonlinear activation-free network – effective against motion blur, compression noise, and film grain at native resolution.",
        /* isDefault   */ true,
        /* minVram     */ 1024
    },
    {
        /* id          */ "clearreality-x4-denoise",
        /* displayName */ "4x-ClearRealityV1 (lightweight denoising + upscale)",
        /* stage       */ StageKind::Denoise,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-ClearRealityV1.onnx",
        /* filename    */ "4x-ClearRealityV1.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Very compact (1.9 MB) 4x upscaler with built-in denoising – fast and VRAM-efficient.",
        /* isDefault   */ false,
        /* minVram     */ 512
    },
    {
        /* id          */ "remacri-x4",
        /* displayName */ "4x Remacri (smooth-texture denoising + upscale)",
        /* stage       */ StageKind::Denoise,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x_foolhardy_Remacri.onnx",
        /* filename    */ "4x_foolhardy_Remacri.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Foolhardy Remacri x4 – cinematic smooth textures with strong noise suppression.",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },

    // ════════════════════════════════════════════════════════════
    // Deblur
    // ════════════════════════════════════════════════════════════
    {
        /* id          */ "realesrnet-deblur-x2",
        /* displayName */ "Real-ESRNet x2+ (MiGraphX-safe deblur + restore)",
        /* stage       */ StageKind::Deblur,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 2,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/2x_ModernSpanimationV2_clamp_op20_onnxslim.onnx",
        /* filename    */ "2x_ModernSpanimationV2_clamp_op20_onnxslim.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Compile-proven x2 restoration model used as the default MiGraphX deblur option. Conservative sharpening and stable reconstruction make it a safer substitute for hostile native deblur exports.",
        /* isDefault   */ true,
        /* minVram     */ 512
    },
    {
        /* id          */ "openproteus-deblur-x2",
        /* displayName */ "OpenProteus Compact x2 (detail-focused deblur)",
        /* stage       */ StageKind::Deblur,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 2,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/Sirosky/Upscale-Hub/releases/download/OpenProteus/2x_OpenProteus_Compact_i2_70K_fp32.onnx",
        /* filename    */ "2x_OpenProteus_Compact_i2_70K_fp32.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Compile-proven x2 OpenProteus export tuned for restrained sharpening and detail recovery. Best suited when the deblur stage should recover crispness without aggressive GAN-style overshoot.",
        /* isDefault   */ false,
        /* minVram     */ 512
    },
    {
        /* id          */ "clearreality-deblur-x4",
        /* displayName */ "ClearReality V1 x4 (fast deblur + upscale)",
        /* stage       */ StageKind::Deblur,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-ClearRealityV1.onnx",
        /* filename    */ "4x-ClearRealityV1.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Fastest compile-proven MiGraphX-compatible deblur replacement in the repo history. Lightweight x4 restoration that trades peak fidelity for predictable compilation and throughput.",
        /* isDefault   */ false,
        /* minVram     */ 256
    },

    // ════════════════════════════════════════════════════════════
    // Dehalo
    // ════════════════════════════════════════════════════════════
    {
        /* id          */ "nafnet-dehalo",
        /* displayName */ "NAFNet Halo Suppression (blind restoration)",
        /* stage       */ StageKind::Dehalo,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/opencv/deblurring_nafnet/resolve/main/deblurring_nafnet_2025may.onnx",
        /* filename    */ "deblurring_nafnet_2025may.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "NAFNet blind restoration suppresses high-frequency ringing and halos at native resolution.",
        /* isDefault   */ true,
        /* minVram     */ 1024
    },

    // ════════════════════════════════════════════════════════════
    // Color Fix
    // ════════════════════════════════════════════════════════════
    {
        /* id          */ "iqa-color-enhance",
        /* displayName */ "Color Correction (parametric)",
        /* stage       */ StageKind::ColorFix,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "",
        /* filename    */ "",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Parametric color correction using tunable contrast, brightness, saturation, gamma, and vibrance sliders.",
        /* isDefault   */ true,
        /* minVram     */ 0
    },
    {
        /* id          */ "swinir-color",
        /* displayName */ "SwinIR x4 Color Enhancement (ONNX)",
        /* stage       */ StageKind::ColorFix,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/rocca/swin-ir-onnx/resolve/main/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx",
        /* filename    */ "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "SwinIR-M real super-resolution (BSRGAN degradation) – x4 color restoration and fidelity enhancement.",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },

    // ════════════════════════════════════════════════════════════
    // Upscale
    // ════════════════════════════════════════════════════════════
    {
        /* id          */ "realesrgan-x4-general",
        /* displayName */ "Real-ESRGAN x4 General (ONNX)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/RealESRGAN_x4.onnx",
        /* filename    */ "RealESRGAN_x4.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Real-ESRGAN x4plus upscaling for general real-world photography and video.",
        /* isDefault   */ true,
        /* minVram     */ 2048
    },
    {
        /* id          */ "openproteus-compact-x2",
        /* displayName */ "OpenProteus Compact x2 (ONNX)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 2,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/Sirosky/Upscale-Hub/releases/download/OpenProteus/2x_OpenProteus_Compact_i2_70K_fp32.onnx",
        /* filename    */ "2x_OpenProteus_Compact_i2_70K_fp32.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Compact OpenProteus upscale model for clean live-action HD/FHD footage, tuned for restrained sharpening and faithful 2x enlargement.",
        /* isDefault   */ false,
        /* minVram     */ 512
    },
    {
        /* id          */ "realesrgan-x4plus-pth",
        /* displayName */ "Real-ESRGAN x4+ (PyTorch .pth)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Pytorch,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        /* filename    */ "RealESRGAN_x4plus.pth",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Original PyTorch checkpoint for Real-ESRGAN x4plus. Compiled through torch export (PyTorch -> ONNX -> MiGraphX).",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },
    {
        /* id          */ "animesharp-x4",
        /* displayName */ "4x-AnimeSharp (anime + CGI upscale)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-AnimeSharp.onnx",
        /* filename    */ "4x-AnimeSharp.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "AnimeSharp x4 – fine-tuned for anime, cartoon, and CGI content with crisp line detail.",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },
    {
        /* id          */ "ultrasharpv2-x4",
        /* displayName */ "4x-UltraSharpV2 (photo / video upscale)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-UltraSharpV2.onnx",
        /* filename    */ "4x-UltraSharpV2.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "4x-UltraSharpV2 – highly rated general-purpose 4x upscaler with excellent sharpness.",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },
    {
        /* id          */ "realesrgan-x4-axera",
        /* displayName */ "Real-ESRGAN x4 (AXERA-TECH ONNX export)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/AXERA-TECH/Real-ESRGAN/resolve/main/onnx/realesrgan-x4.onnx",
        /* filename    */ "realesrgan-x4-axera.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Official AXERA-TECH dynamic-resolution ONNX export of Real-ESRGAN x4plus.",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },
    {
        /* id          */ "swinir-x4-general",
        /* displayName */ "SwinIR x4 General (Transformer SR, ONNX)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/rocca/swin-ir-onnx/resolve/main/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx",
        /* filename    */ "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Swin Transformer-based x4 super-resolution with BSRGAN degradation training.",
        /* isDefault   */ false,
        /* minVram     */ 4096
    },
    {
        /* id          */ "modernspanimation-x2",
        /* displayName */ "ModernSpanimation v2 x2 fp16 (anime, ONNX)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp16,
        /* scale       */ 2,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/2x_ModernSpanimationV2_clamp_op20_fp16_onnxslim.onnx",
        /* filename    */ "2x_ModernSpanimationV2_clamp_op20_fp16_onnxslim.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "ModernSpanimation v2 fp16 slim ONNX – fp16-quantised export, half the file size and VRAM of fp32; optimised for anime and animated video at x2.",
        /* isDefault   */ false,
        /* minVram     */ 256
    },

    // ── Lightweight / fast-enough 2× general ─────────────────────
    {
        /* id          */ "modernspanimation-x2-fp32",
        /* displayName */ "ModernSpanimation v1 x2 fp32 (lightweight general, ONNX)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 2,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/2x_ModernSpanimationV1_fp32_op17.onnx",
        /* filename    */ "2x_ModernSpanimationV1_fp32_op17.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "ModernSpanimation v1 fp32 – compact 2.8 MB model; ideal for 1080p→1440p or any moderate upscale "
                          "without heavy VRAM use. Fast enough for near-real-time batch processing.",
        /* isDefault   */ false,
        /* minVram     */ 256
    },
    {
        /* id          */ "realesrnet-x2plus",
        /* displayName */ "ModernSpanimation v2 x2 fp32 slim (conservative 2x, ONNX)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 2,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/2x_ModernSpanimationV2_clamp_op20_onnxslim.onnx",
        /* filename    */ "2x_ModernSpanimationV2_clamp_op20_onnxslim.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "RealESRNet x2+ without the GAN discriminator – stable, artifact-free x2 upscale at roughly half "
                          "the VRAM of the full x4plus GAN model. Good choice for 720p→1080p or 1080p→1440p.",
        /* isDefault   */ false,
        /* minVram     */ 512
    },

    // ── Lightweight / fast-enough 4× options ─────────────────────
    {
        /* id          */ "clearreality-x4-fast",
        /* displayName */ "4x-ClearRealityV1 — Ultra-Fast Lightweight (ONNX)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-ClearRealityV1.onnx",
        /* filename    */ "4x-ClearRealityV1.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "ClearRealityV1 x4 – 1.9 MB model file; fastest 4x upscaler in the catalog. Trades some fine "
                          "texture fidelity for dramatically lower VRAM and inference time. Good for 1080p→4K when "
                          "\"good enough\" quality is acceptable.",
        /* isDefault   */ false,
        /* minVram     */ 256
    },
    {
        /* id          */ "ultrasharpv2-lite-x4",
        /* displayName */ "4x-UltraSharpV2 Lite (conservative 4x, ONNX)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-UltraSharpV2_Lite.onnx",
        /* filename    */ "4x-UltraSharpV2_Lite.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "RealESRNet x4+ trained without a GAN discriminator – produces slightly softer but more stable "
                          "results than the full GAN-trained x4plus. Lower chance of hallucinated textures on real-world video.",
        /* isDefault   */ false,
        /* minVram     */ 1024
    },
    {
        /* id          */ "realistic-rescaler-x4",
        /* displayName */ "4x RealisticRescaler (fast general, ONNX)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x_RealisticRescaler_100000_G.onnx",
        /* filename    */ "4x_RealisticRescaler_100000_G.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "RealisticRescaler x4 – general-purpose model biased towards realism. Good detail recovery and "
                          "natural textures. Suitable when throughput matters more than maximum fidelity.",
        /* isDefault   */ false,
        /* minVram     */ 1024
    },
    {
        /* id          */ "realesrgan-x4plus-ncnn",
        /* displayName */ "Real-ESRGAN x4+ (NCNN Vulkan)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::NcnnBin,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        /* filename    */ "realesrgan-x4plus.param",
        /* dlUrlAux    */ "",
        /* filenameAux */ "realesrgan-x4plus.bin",
        /* description */ "Official Real-ESRGAN x4plus NCNN Vulkan model extracted from the upstream portable Linux bundle.",
        /* isDefault   */ false,
        /* minVram     */ 1024,
        /* archiveSubPath */ "models/realesrgan-x4plus.param",
        /* archiveSubPathAux */ "models/realesrgan-x4plus.bin"
    },
    {
        /* id          */ "realesrgan-x4plus-anime-ncnn",
        /* displayName */ "Real-ESRGAN x4+ Anime (NCNN Vulkan)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::NcnnBin,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        /* filename    */ "realesrgan-x4plus-anime.param",
        /* dlUrlAux    */ "",
        /* filenameAux */ "realesrgan-x4plus-anime.bin",
        /* description */ "Official anime-tuned Real-ESRGAN NCNN Vulkan model from the upstream portable Linux bundle.",
        /* isDefault   */ false,
        /* minVram     */ 1024,
        /* archiveSubPath */ "models/realesrgan-x4plus-anime.param",
        /* archiveSubPathAux */ "models/realesrgan-x4plus-anime.bin"
    },
    {
        /* id          */ "realesr-animevideov3-x2-ncnn",
        /* displayName */ "Real-ESRGAN AnimeVideo v3 x2 (NCNN Vulkan)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::NcnnBin,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 2,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        /* filename    */ "realesr-animevideov3-x2.param",
        /* dlUrlAux    */ "",
        /* filenameAux */ "realesr-animevideov3-x2.bin",
        /* description */ "Official Real-ESRGAN AnimeVideo v3 x2 NCNN Vulkan model for fast anime and animation upscaling.",
        /* isDefault   */ false,
        /* minVram     */ 512,
        /* archiveSubPath */ "models/realesr-animevideov3-x2.param",
        /* archiveSubPathAux */ "models/realesr-animevideov3-x2.bin"
    },
    {
        /* id          */ "realesr-animevideov3-x3-ncnn",
        /* displayName */ "Real-ESRGAN AnimeVideo v3 x3 (NCNN Vulkan)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::NcnnBin,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 3,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        /* filename    */ "realesr-animevideov3-x3.param",
        /* dlUrlAux    */ "",
        /* filenameAux */ "realesr-animevideov3-x3.bin",
        /* description */ "Official Real-ESRGAN AnimeVideo v3 x3 NCNN Vulkan model for animation-heavy video.",
        /* isDefault   */ false,
        /* minVram     */ 768,
        /* archiveSubPath */ "models/realesr-animevideov3-x3.param",
        /* archiveSubPathAux */ "models/realesr-animevideov3-x3.bin"
    },
    {
        /* id          */ "realesr-animevideov3-x4-ncnn",
        /* displayName */ "Real-ESRGAN AnimeVideo v3 x4 (NCNN Vulkan)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::NcnnBin,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        /* filename    */ "realesr-animevideov3-x4.param",
        /* dlUrlAux    */ "",
        /* filenameAux */ "realesr-animevideov3-x4.bin",
        /* description */ "Official Real-ESRGAN AnimeVideo v3 x4 NCNN Vulkan model for aggressive anime upscaling.",
        /* isDefault   */ false,
        /* minVram     */ 1024,
        /* archiveSubPath */ "models/realesr-animevideov3-x4.param",
        /* archiveSubPathAux */ "models/realesr-animevideov3-x4.bin"
    },

    // ── Lightweight 2× anime ──────────────────────────────────────
    {
        /* id          */ "modernspanimation-x2-v1compact",
        /* displayName */ "ModernSpanimation v1 Compact x2 (anime, ONNX)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 2,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/2x_ModernSpanimationV2_clamp_op20.onnx",
        /* filename    */ "2x_ModernSpanimationV2_clamp_op20.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "ModernSpanimation v1 Compact – significantly smaller network than v2 Full for anime x2 upscale. "
                          "Best pick when speed is the priority for animated content.",
        /* isDefault   */ false,
        /* minVram     */ 256
    },
    {
        /* id          */ "animejananai-hd-compact-x2",
        /* displayName */ "AnimeJaNai HD V3 Sharp Compact x2 (anime, PyTorch)",
        /* stage       */ StageKind::Upscale,
        /* format      */ ModelFormat::Pytorch,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 2,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth",
        /* filename    */ "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "AnimeJaNai HD V3 Sharp Compact – purpose-built for 1080p anime upscale to 2160p with a compact "
                          "network. PyTorch format – will be exported to ONNX then compiled to MXR.",
        /* isDefault   */ false,
        /* minVram     */ 512
    },

    // ════════════════════════════════════════════════════════════
    // Sharpen
    // ════════════════════════════════════════════════════════════
    {
        /* id          */ "sharpen-cas",
        /* displayName */ "Contrast-Adaptive Sharpening (CAS)",
        /* stage       */ StageKind::Sharpen,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "",
        /* filename    */ "",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "AMD FidelityFX CAS + unsharp mask pipeline via FFmpeg – no model download needed.",
        /* isDefault   */ true,
        /* minVram     */ 0
    },

    // ════════════════════════════════════════════════════════════
    // Stereo 3D / 2D -> 3D SBS
    // ════════════════════════════════════════════════════════════
    {
        /* id          */ "depth-anything-v2-small-fp16",
        /* displayName */ "Depth Anything V2 Small fp16 (2D -> 3D depth, ONNX)",
        /* stage       */ StageKind::Stereo3D,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp16,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx",
        /* filename    */ "depth-anything-v2-small-fp16.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Depth Anything V2 Small ONNX export. Fast relative-depth model suitable for 2D-to-3D SBS generation with iw3-style divergence and convergence controls.",
        /* isDefault   */ true,
        /* minVram     */ 1024,
        /* archiveSubPath */ "",
        /* archiveSubPathAux */ "",
        /* migraphxCompileWidth */ 0,
        /* migraphxCompileHeight */ 0,
        /* migraphxOnnxTransform */ MiGraphXOnnxTransform::ResizeCubicToLinear
    },
    {
        /* id          */ "depth-anything-v2-base-fp16",
        /* displayName */ "Depth Anything V2 Base fp16 (2D -> 3D depth, ONNX)",
        /* stage       */ StageKind::Stereo3D,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp16,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/main/onnx/model_fp16.onnx",
        /* filename    */ "depth-anything-v2-base-fp16.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Depth Anything V2 Base fp16 ONNX export. Better scene geometry than the small model at a moderate VRAM cost.",
        /* isDefault   */ false,
        /* minVram     */ 2048,
        /* archiveSubPath */ "",
        /* archiveSubPathAux */ "",
        /* migraphxCompileWidth */ 0,
        /* migraphxCompileHeight */ 0,
        /* migraphxOnnxTransform */ MiGraphXOnnxTransform::ResizeCubicToLinear
    },
    {
        /* id          */ "depth-anything-v2-large-fp16",
        /* displayName */ "Depth Anything V2 Large fp16 (2D -> 3D depth, ONNX)",
        /* stage       */ StageKind::Stereo3D,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp16,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model_fp16.onnx",
        /* filename    */ "depth-anything-v2-large-fp16.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Depth Anything V2 Large fp16 ONNX export. Highest-detail Depth Anything V2 variant in the built-in stereo catalog.",
        /* isDefault   */ false,
        /* minVram     */ 4096,
        /* archiveSubPath */ "",
        /* archiveSubPathAux */ "",
        /* migraphxCompileWidth */ 0,
        /* migraphxCompileHeight */ 0,
        /* migraphxOnnxTransform */ MiGraphXOnnxTransform::ResizeCubicToLinear
    },
    {
        /* id          */ "distill-any-depth-small",
        /* displayName */ "Distill Any Depth Small (2D -> 3D depth, ONNX)",
        /* stage       */ StageKind::Stereo3D,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/FuryTMP/Distill-Any-Depth-Small-onnx/resolve/main/Distill%20Any%20Depth%20Small/model.onnx",
        /* filename    */ "distill-any-depth-small.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Distill Any Depth Small ONNX export. Compact distilled depth model for fast 2D-to-3D conversion; the source export is fp32, but the app compiles fp16 artifacts by default.",
        /* isDefault   */ false,
        /* minVram     */ 1536,
        /* archiveSubPath */ "",
        /* archiveSubPathAux */ "",
        /* migraphxCompileWidth */ 0,
        /* migraphxCompileHeight */ 0
    },
    {
        /* id          */ "distill-any-depth-base",
        /* displayName */ "Distill Any Depth Base (2D -> 3D depth, ONNX)",
        /* stage       */ StageKind::Stereo3D,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/FuryTMP/Distill-Any-Depth-Base-onnx/resolve/main/Distill%20Any%20Depth%20Base/model.onnx",
        /* filename    */ "distill-any-depth-base.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Distill Any Depth Base ONNX export. Balanced distilled depth estimator for stereo synthesis; the source export is fp32, but the app compiles fp16 artifacts by default.",
        /* isDefault   */ false,
        /* minVram     */ 3072,
        /* archiveSubPath */ "",
        /* archiveSubPathAux */ "",
        /* migraphxCompileWidth */ 0,
        /* migraphxCompileHeight */ 0
    },
    {
        /* id          */ "distill-any-depth-large",
        /* displayName */ "Distill Any Depth Large (2D -> 3D depth, ONNX)",
        /* stage       */ StageKind::Stereo3D,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/FuryTMP/Distill-Any-Depth-Large-onnx/resolve/main/Distill%20Any%20Depth%20Large/model.onnx",
        /* filename    */ "distill-any-depth-large.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Distill Any Depth Large ONNX export. Highest-capacity distilled monocular depth model in the built-in stereo catalog; the source export is fp32, but the app compiles fp16 artifacts by default.",
        /* isDefault   */ false,
        /* minVram     */ 6144,
        /* archiveSubPath */ "",
        /* archiveSubPathAux */ "",
        /* migraphxCompileWidth */ 0,
        /* migraphxCompileHeight */ 0
    },

    // ════════════════════════════════════════════════════════════
    // Interpolate (Frame Interpolation)
    // ════════════════════════════════════════════════════════════
    {
        /* id          */ "rife-v4-7",
        /* displayName */ "RIFE v4.7 (2x frame interpolation, ONNX)",
        /* stage       */ StageKind::Interpolate,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 2.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/rife47_ensemble_True_scale_1_sim.onnx",
        /* filename    */ "rife47_ensemble_True_scale_1_sim.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "RIFE v4.7 optical-flow frame interpolation – ensemble mode for smoother results.",
        /* isDefault   */ true,
        /* minVram     */ 2048
    },
    {
        /* id          */ "rife-v4-8",
        /* displayName */ "RIFE v4.8 (2x frame interpolation, ONNX)",
        /* stage       */ StageKind::Interpolate,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 2.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/rife48_ensemble_True_scale_1_sim.onnx",
        /* filename    */ "rife48_ensemble_True_scale_1_sim.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "RIFE v4.8 optical-flow frame interpolation – incremental quality improvement over v4.7.",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },
    {
        /* id          */ "rife-v4-9",
        /* displayName */ "RIFE v4.9 (2x frame interpolation, ONNX)",
        /* stage       */ StageKind::Interpolate,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 2.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/rife49_ensemble_True_scale_1_sim.onnx",
        /* filename    */ "rife49_ensemble_True_scale_1_sim.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "RIFE v4.9 optical-flow frame interpolation – latest stable ensemble version.",
        /* isDefault   */ false,
        /* minVram     */ 2048
    },
    {
        /* id          */ "interp-ffmpeg",
        /* displayName */ "FFmpeg minterpolate (CPU fallback)",
        /* stage       */ StageKind::Interpolate,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 2.0,
        /* downloadUrl */ "",
        /* filename    */ "",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "Software motion-compensated interpolation via FFmpeg minterpolate – no GPU needed.",
        /* isDefault   */ false,
        /* minVram     */ 0
    }
};
// ─────────────────────────────────────────────────────────────────

const std::vector<ModelEntry>& builtinModelCatalog() {
    return kCatalog;
}

std::vector<const ModelEntry*> catalogEntriesForStage(StageKind stage) {
    std::vector<const ModelEntry*> result;
    for (const auto& entry : kCatalog) {
        if (entry.stage == stage) {
            result.push_back(&entry);
        }
    }
    return result;
}

const ModelEntry* catalogEntryById(const std::string& id) {
    for (const auto& entry : kCatalog) {
        if (entry.id == id) {
            return &entry;
        }
    }
    return nullptr;
}

std::string inferModelIdFromPath(const std::string& path) {
    if (path.empty()) {
        return {};
    }

    auto candidate = std::filesystem::path(path).stem().string();
    if (candidate.empty()) {
        return {};
    }

    if (const auto* entry = catalogEntryById(candidate); entry != nullptr) {
        return entry->id;
    }

    const auto trimSuffix = [&](const std::string& suffix) {
        if (candidate.size() > suffix.size() &&
            candidate.compare(candidate.size() - suffix.size(), suffix.size(), suffix) == 0) {
            candidate.resize(candidate.size() - suffix.size());
            return true;
        }
        return false;
    };

    trimSuffix("_fp16");
    trimSuffix("_fp32");
    trimSuffix("_int8");
    if (const auto* entry = catalogEntryById(candidate); entry != nullptr) {
        return entry->id;
    }

    auto isDigits = [](const std::string& value) {
        return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return std::isdigit(ch) != 0;
        });
    };

    while (true) {
        const std::size_t lastUnderscore = candidate.rfind('_');
        if (lastUnderscore == std::string::npos) {
            break;
        }
        const std::string suffix = candidate.substr(lastUnderscore + 1);
        bool stripped = false;
        if ((suffix.size() >= 2 && suffix[0] == 'b' && isDigits(suffix.substr(1))) ||
            suffix == "fp16" || suffix == "fp32" || suffix == "int8") {
            stripped = true;
        } else {
            const std::size_t xPos = suffix.find('x');
            if (xPos != std::string::npos &&
                isDigits(suffix.substr(0, xPos)) &&
                isDigits(suffix.substr(xPos + 1))) {
                stripped = true;
            }
        }
        if (!stripped) {
            break;
        }
        candidate.resize(lastUnderscore);
        if (const auto* entry = catalogEntryById(candidate); entry != nullptr) {
            return entry->id;
        }
    }

    return {};
}

std::string modelFamilyId(const ModelEntry& entry) {
    if (!entry.familyId.empty()) {
        return entry.familyId;
    }
    if (const auto* builtin = builtinFamilyInfoForId(entry.id); builtin != nullptr) {
        return builtin->id;
    }
    return entry.id;
}

std::string modelFamilyName(const ModelEntry& entry) {
    if (!entry.familyName.empty()) {
        return entry.familyName;
    }
    if (const auto* builtin = builtinFamilyInfoForId(entry.id); builtin != nullptr) {
        return builtin->name;
    }
    return entry.displayName;
}

std::vector<StageKind> modelCapabilities(const ModelEntry& entry) {
    if (!entry.capabilities.empty()) {
        return entry.capabilities;
    }

    std::vector<StageKind> out = {entry.stage};
    const std::string familyId = modelFamilyId(entry);
    std::unordered_set<int> seen = {static_cast<int>(entry.stage)};
    for (const auto& candidate : kCatalog) {
        if (candidate.id == entry.id) {
            continue;
        }
        if (modelFamilyId(candidate) != familyId) {
            continue;
        }
        if (seen.insert(static_cast<int>(candidate.stage)).second) {
            out.push_back(candidate.stage);
        }
    }
    return out;
}

bool modelSupportsCapability(const ModelEntry& entry, const StageKind capability) {
    const auto capabilities = modelCapabilities(entry);
    return std::find(capabilities.begin(), capabilities.end(), capability) != capabilities.end();
}

bool modelCanFuseRequestedCapabilities(const ModelEntry& entry,
                                       const std::vector<StageKind>& requested) {
    if (requested.size() <= 1) {
        return true;
    }

    bool supportsFusedExecution = entry.supportsFusedExecution;
    bool supportsSelectiveCapabilities = entry.supportsSelectiveCapabilities;
    if (const auto* builtin = builtinFamilyInfoForId(entry.id); builtin != nullptr) {
        supportsFusedExecution = supportsFusedExecution || builtin->supportsFusedExecution;
        supportsSelectiveCapabilities =
            supportsSelectiveCapabilities || builtin->supportsSelectiveCapabilities;
    }
    if (!supportsFusedExecution) {
        return false;
    }

    const auto capabilities = modelCapabilities(entry);
    std::unordered_set<int> supported;
    for (const auto kind : capabilities) {
        supported.insert(static_cast<int>(kind));
    }
    for (const auto kind : requested) {
        if (supported.find(static_cast<int>(kind)) == supported.end()) {
            return false;
        }
    }
    if (supportsSelectiveCapabilities) {
        return true;
    }
    return requested.size() == capabilities.size();
}

bool modelLooksAnimationFocused(const ModelEntry& entry) {
    auto containsKeyword = [](const std::string& haystack,
                              const char* needle) {
        if (needle == nullptr || *needle == '\0') {
            return false;
        }
        std::string lowered;
        lowered.reserve(haystack.size());
        for (const char ch : haystack) {
            lowered.push_back(
                static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
        return lowered.find(needle) != std::string::npos;
    };

    return containsKeyword(entry.id, "anime")
        || containsKeyword(entry.id, "animation")
        || containsKeyword(entry.displayName, "anime")
        || containsKeyword(entry.displayName, "animation")
        || containsKeyword(entry.description, "anime")
        || containsKeyword(entry.description, "animation")
        || containsKeyword(entry.description, "animated")
        || containsKeyword(entry.description, "cartoon")
        || containsKeyword(entry.description, "cgi");
}

bool modelSupportsBackend(const ModelEntry& entry, const BackendType backend) {
    switch (backend) {
        case BackendType::NcnnVulkan:
            return entry.sourceFormat == ModelFormat::NcnnBin;
        case BackendType::MiGraphX:
            return entry.sourceFormat == ModelFormat::Onnx
                || entry.sourceFormat == ModelFormat::Pytorch;
        default:
            return true;
    }
}

const ModelEntry* preferredBackendModelForStage(const StageKind stage,
                                                const BackendType backend) {
    const auto entries = catalogEntriesForStage(stage);
    if (entries.empty()) {
        return nullptr;
    }

    const auto pickMatching = [&](const auto& predicate) -> const ModelEntry* {
        for (const auto* entry : entries) {
            if (entry != nullptr && predicate(*entry)) {
                return entry;
            }
        }
        return nullptr;
    };

    if (const auto* preferred = pickMatching([backend](const ModelEntry& entry) {
            return entry.isDefault && modelSupportsBackend(entry, backend);
        })) {
        return preferred;
    }

    if (const auto* preferred = pickMatching([backend](const ModelEntry& entry) {
            return modelSupportsBackend(entry, backend)
                && !modelLooksAnimationFocused(entry);
        })) {
        return preferred;
    }

    if (const auto* preferred = pickMatching([backend](const ModelEntry& entry) {
            return modelSupportsBackend(entry, backend);
        })) {
        return preferred;
    }

    return nullptr;
}

}  // namespace ave
