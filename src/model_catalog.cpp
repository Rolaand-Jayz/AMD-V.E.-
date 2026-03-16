#include "ave/model_catalog.hpp"

#include <algorithm>

namespace ave {

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
        /* id          */ "nafnet-deblur-gopro",
        /* displayName */ "NAFNet GoPro Motion Deblur (ONNX)",
        /* stage       */ StageKind::Deblur,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 1,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/opencv/deblurring_nafnet/resolve/main/deblurring_nafnet_2025may.onnx",
        /* filename    */ "deblurring_nafnet_2025may.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "NAFNet trained on GoPro – strong motion blur removal at native resolution. Official OpenCV Zoo model.",
        /* isDefault   */ true,
        /* minVram     */ 1024
    },
    {
        /* id          */ "ultrasharpv2-deblur",
        /* displayName */ "4x-UltraSharpV2 (deblur + upscale)",
        /* stage       */ StageKind::Deblur,
        /* format      */ ModelFormat::Onnx,
        /* precision   */ ModelPrecision::Fp32,
        /* scale       */ 4,
        /* fpsMul      */ 1.0,
        /* downloadUrl */ "https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/4x-UltraSharpV2.onnx",
        /* filename    */ "4x-UltraSharpV2.onnx",
        /* dlUrlAux    */ "",
        /* filenameAux */ "",
        /* description */ "4x-UltraSharpV2 – second-generation UltraSharp with improved deblurring and sharpness recovery.",
        /* isDefault   */ false,
        /* minVram     */ 2048
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

}  // namespace ave
