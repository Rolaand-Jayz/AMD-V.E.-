# AGENTS.md

Guide for AI agents working in the AMD Video Enhancer C++ codebase.

## Project Overview

AMD-exclusive C++ video enhancement pipeline. No Python, no CUDA, no NVIDIA.

- **Primary backend**: MiGraphX (ROCm) - ONNX load, compile, GPU inference
- **Vulkan compute backend**: Vulkan SPIR-V compute shaders for GPU processing
- **Fallback backend**: NCNN Vulkan - Vulkan-accelerated inference
- **Always available**: FFmpeg filter chain - no model required

## Essential Commands

### Build

**IMPORTANT**: Always enable ALL available backends. Never build without them.

```bash
# Standard build — ALL backends enabled (this is the ONLY build you should run)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAVE_HAVE_CURL=ON \
  -DAVE_HAVE_MIGRAPHX=ON \
  -DAVE_HAVE_HIP=ON \
  -DAVE_HAVE_VULKAN=ON \
  -DAVE_HAVE_NCNN=ON
cmake --build build -j
```

> **Never** use a bare `cmake -S . -B build` without backend flags.
> MiGraphX, HIP, Vulkan, and NCNN must always be explicitly enabled.

### Test

```bash
ctest --test-dir build --output-on-failure
```

### Run

```bash
# CLI
./build/ave --input input.mp4 --output output.mp4 --backend auto --stage upscale:width=3840,height=2160

# GUI
./build/ave_gui

# List available backends
./build/ave --list-backends

# Dry-run (plan only)
./build/ave --input input.mp4 --output output.mp4 --stage sharpen --dry-run
```

## Build Targets

| Target | Description |
|--------|-------------|
| `ave` | CLI executable |
| `ave_gui` | Qt6 GUI executable (optional, requires Qt6) |
| `ave_core` | Core static library |
| `planner_tests` | Test executable |

## CMake Options

| Option               | Default | Description                                                    |
| -------------------- | ------- | -------------------------------------------------------------- |
| `AVE_STRICT_WARNINGS`| ON      | Enable `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion` |
| `AVE_BUILD_GUI`      | ON      | Build Qt6 GUI                                                  |
| `AVE_HAVE_CURL`      | ON      | Enable libcurl for model downloads                             |
| `AVE_HAVE_VULKAN`    | ON      | Enable Vulkan runtime (compute shaders, interop)               |
| `AVE_HAVE_MIGRAPHX`  | OFF     | Enable MiGraphX ROCm runtime — **always pass ON**              |
| `AVE_HAVE_HIP`       | OFF     | Enable HIP GPU headers (ROCm) — **always pass ON**             |
| `AVE_HAVE_NCNN`      | OFF     | Enable NCNN Vulkan runtime — **always pass ON**                |
| `AVE_HAVE_ROCTX`     | OFF     | Enable ROCTx markers for rocprof tracing                       |

> CMake defaults for MIGRAPHX, HIP, and NCNN are OFF in `CMakeLists.txt`, but
> **agents must always explicitly set them ON** in the cmake configure command.
> The standard build command above already does this.

## Code Organization

```text
├── include/ave/           # Public headers (namespace ave)
│   ├── backends/          # Backend-specific headers
│   │   ├── glsl_shader_backend.hpp
│   │   ├── migraphx_backend.hpp
│   │   ├── ncnn_vulkan_backend.hpp
│   │   ├── vapoursynth_backend.hpp
│   │   └── vulkan_compute_backend.hpp
│   ├── app_settings.hpp
│   ├── backend.hpp        # IAcceleratorBackend interface
│   ├── backend_manager.hpp
│   ├── cli.hpp
│   ├── error_taxonomy.hpp
│   ├── ffmpeg_runner.hpp
│   ├── filter_catalog.hpp
│   ├── frame_io.hpp
│   ├── interop_bridge.hpp
│   ├── job.hpp            # VideoJob, EncodeSettings structs
│   ├── model_catalog.hpp
│   ├── model_manager.hpp
│   ├── observability.hpp
│   ├── planner.hpp        # PipelinePlanner class
│   ├── scene_detector.hpp
│   ├── stage.hpp          # EnhancementStage, ParameterValue
│   ├── tensor_contract.hpp
│   ├── types.hpp          # StageKind enum
│   ├── video_processor.hpp
│   └── vulkan_runtime.hpp
├── src/
│   ├── backends/          # Backend implementations
│   │   ├── glsl_shader_backend.cpp
│   │   ├── migraphx_backend.cpp
│   │   ├── ncnn_vulkan_backend.cpp
│   │   ├── vapoursynth_backend.cpp
│   │   └── vulkan_compute_backend.cpp
│   ├── gui/               # Qt6 GUI (conditional)
│   │   ├── filter_browser.cpp/hpp
│   │   ├── main_gui.cpp
│   │   ├── main_window.cpp/hpp
│   │   ├── model_manager_dialog.cpp/hpp
│   │   ├── settings_dialog.cpp/hpp
│   │   └── toggle_switch.cpp/hpp
│   ├── app_settings.cpp
│   ├── backend.cpp
│   ├── backend_manager.cpp
│   ├── cli.cpp
│   ├── error_taxonomy.cpp
│   ├── ffmpeg_runner.cpp
│   ├── filter_catalog.cpp
│   ├── frame_io.cpp
│   ├── frame_io_vulkan.cpp
│   ├── interop_bridge.cpp
│   ├── main.cpp
│   ├── model_catalog.cpp
│   ├── model_manager.cpp
│   ├── observability.cpp
│   ├── planner.cpp
│   ├── scene_detector.cpp
│   ├── stage.cpp
│   ├── tensor_contract.cpp
│   ├── types.cpp
│   ├── video_processor.cpp
│   └── vulkan_runtime.cpp
├── tests/
│   └── planner_tests.cpp  # Unit tests
└── docs/
    ├── FEATURE_PARITY_MATRIX.md
    ├── GOLD_STANDARD_FOR_IMPLEMENTATION.md
    ├── PARITY_PLAN.md
    └── migraphx_debugging_playbook.md
```

## Naming Conventions

- **Headers**: `#pragma once` guards, `ave/xxx.hpp` paths
- **Namespace**: All code in `namespace ave { ... }`
- **Types**: PascalCase (e.g., `StageKind`, `EnhancementStage`, `PipelinePlanner`)
- **Functions/Methods**: camelCase (e.g., `parseCli`, `runStage`, `buildUi`)
- **Private members**: `name_` suffix (e.g., `inputPathEdit_`, `impl_`)
- **Enums**: `EnumClass` with PascalCase values (e.g., `StageKind::RestoreCompression`)
- **Internal helpers**: Anonymous namespace `namespace { ... }`

## Code Patterns

### Headers

```cpp
#pragma once

#include <string>
#include "ave/types.hpp"

namespace ave {

class MyClass {
  public:
    explicit MyClass();
    ~MyClass() = default;
  private:
    int value_ = 0;
};

}  // namespace ave
```

### Error Handling

- Use output string parameters: `bool doSomething(std::string& error)`
- Return `std::optional<T>` for fallible value returns
- Check availability before use: `bool isAvailable(std::string& reason) const`

### Conditional Compilation

Backend code uses preprocessor guards:

```cpp
#ifdef AVE_HAVE_NCNN
#  include <ncnn/ncnn/net.h>
#endif

#ifdef AVE_HAVE_NCNN
    // Full implementation
#else
    // Stub/fallback implementation
#endif
```

### PIMPL Pattern

Backends use PIMPL to hide implementation details:

```cpp
class NcnnVulkanBackend {
  public:
    NcnnVulkanBackend();
    ~NcnnVulkanBackend();
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

## Enhancement Stages

Stages execute in deterministic order (enforced by `PipelinePlanner`):

1. **Group 0**: `RestoreCompression`, `RemoveArtifacts`, `Denoise`, `Deblur`, `Dehalo`
2. **Group 1**: `ColorFix`
3. **Group 2**: `Upscale`
4. **Group 3**: `Sharpen`
5. **Group 4**: `Interpolate` (always last)

### Stage Aliases (CLI)

| Stage                   | Aliases                    |
| ----------------------- | -------------------------- |
| `restore_compression`   | `decompress`, `deh264`     |
| `remove_artifacts`      | `deartifact`, `deblock`    |

## Backend Interface

All accelerators implement `IAcceleratorBackend`:

```cpp
class IAcceleratorBackend {
  public:
    virtual ~IAcceleratorBackend() = default;
    virtual BackendType type() const = 0;
    virtual std::string name() const = 0;
    virtual bool isAvailable(std::string& reason) const = 0;
    virtual bool initialize(std::string& error) = 0;
    virtual StageResult runStage(const EnhancementStage& stage, std::string& error) = 0;
    virtual StageResult processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts = {}) = 0;
};
```

Backend priority when `--backend auto`:

1. MiGraphX (ROCm) - requires `/opt/rocm`, `rocminfo`, `libmigraphx.so`
2. Vulkan Compute - requires Vulkan SDK and `libvulkan.so`
3. NCNN (Vulkan) - requires `vulkaninfo` or `libvulkan.so`
4. FFmpeg filters - always available

## Model Storage

Models stored in `~/.local/share/ave/models/`:

| Directory     | Contents                              |
| ------------- | ------------------------------------- |
| `downloaded/` | Raw ONNX / PyTorch source files       |
| `migraphx/`   | MiGraphX `.mxr` compiled programs     |
| `optimised/`  | Hardware-optimised compiled programs   |

## Testing

Tests use simple assertion-based approach:

```cpp
#include <cassert>
#include "ave/planner.hpp"

namespace {
void testSomething() {
    PipelinePlanner planner;
    const auto result = planner.plan({...});
    assert(!result.empty());
}
}  // namespace

int main() {
    testSomething();
    return 0;
}
```

## Qt GUI Notes

- Requires Qt 6.2+ with Widgets module
- Uses `Q_OBJECT` macro for signal/slot
- CMake handles MOC/UIC/RCC automatically
- GUI files in `src/gui/` with local includes

## Important Gotchas

1. **Stage ordering is automatic**: User-specified order is ignored; `PipelinePlanner` reorders stages deterministically. Tests verify this behavior.

2. **Backends gracefully degrade**: Missing libraries don't cause build failures; features are disabled via CMake checks.

3. **FFmpeg required at runtime**: `ffmpeg` and `ffprobe` must be in PATH for all operations.

4. **Model path resolution**: Models are resolved via `ModelManager::findModel()` which checks multiple directories.

5. **Thread safety**: Backend implementations use `std::mutex` for model loading; GUI uses `std::atomic<bool>` for run state.

6. **Warning flags are strict by default**: `AVE_STRICT_WARNINGS=ON` enables `-Wconversion -Wsign-conversion` which catch subtle type issues.
