# AGENTS.md

Guide for AI agents working in the AMD Video Enhancer C++ codebase.

## Project Overview

AMD-exclusive C++ video enhancement pipeline. No Python, no CUDA, no NVIDIA.

- **Primary backend**: MiGraphX (ROCm) - ONNX load, compile, GPU inference
- **Fallback backend**: NCNN Vulkan - Vulkan-accelerated inference
- **Always available**: FFmpeg filter chain - no model required

## Essential Commands

### Build

```bash
# Standard build (FFmpeg-only)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Full ROCm build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAVE_HAVE_CURL=ON \
  -DAVE_HAVE_MIGRAPHX=ON \
  -DAVE_HAVE_HIP=ON

# With NCNN Vulkan support
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAVE_HAVE_NCNN=ON
```

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

| Option | Default | Description |
|--------|---------|-------------|
| `AVE_STRICT_WARNINGS` | ON | Enable `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion` |
| `AVE_BUILD_GUI` | ON | Build Qt6 GUI |
| `AVE_HAVE_CURL` | ON | Enable libcurl for model downloads |
| `AVE_HAVE_MIGRAPHX` | OFF | Enable MiGraphX ROCm runtime |
| `AVE_HAVE_HIP` | OFF | Enable HIP GPU headers |
| `AVE_HAVE_NCNN` | OFF | Enable NCNN Vulkan runtime |

## Code Organization

```
├── include/ave/           # Public headers (namespace ave)
│   ├── backends/          # Backend-specific headers
│   ├── backend.hpp        # IAcceleratorBackend interface
│   ├── backend_manager.hpp
│   ├── cli.hpp
│   ├── ffmpeg_runner.hpp
│   ├── job.hpp            # VideoJob, EncodeSettings structs
│   ├── model_catalog.hpp
│   ├── model_manager.hpp
│   ├── planner.hpp        # PipelinePlanner class
│   ├── scene_detector.hpp
│   ├── stage.hpp          # EnhancementStage, ParameterValue
│   ├── types.hpp          # StageKind enum
│   └── video_processor.hpp
├── src/
│   ├── backends/          # Backend implementations
│   │   ├── migraphx_backend.cpp
│   │   └── ncnn_vulkan_backend.cpp
│   ├── gui/               # Qt6 GUI (conditional)
│   │   ├── main_gui.cpp
│   │   ├── main_window.cpp/hpp
│   │   ├── model_manager_dialog.cpp/hpp
│   │   └── toggle_switch.cpp/hpp
│   └── *.cpp              # Core implementations
├── tests/
│   └── planner_tests.cpp  # Unit tests
└── docs/
    └── FEATURE_PARITY_MATRIX.md
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

| Stage | Aliases |
|-------|---------|
| `restore_compression` | `decompress`, `deh264` |
| `remove_artifacts` | `deartifact`, `deblock` |

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
    virtual bool runStage(const EnhancementStage& stage, std::string& error) = 0;
};
```

Backend priority when `--backend auto`:
1. MiGraphX (ROCm) - requires `/opt/rocm`, `rocminfo`, `libmigraphx.so`
2. NCNN (Vulkan) - requires `vulkaninfo` or `libvulkan.so`
3. FFmpeg filters - always available

## Model Storage

Models stored in `~/.local/share/ave/models/`:

| Directory | Contents |
|-----------|----------|
| `downloaded/` | Raw ONNX / PyTorch source files |
| `migraphx/` | MiGraphX `.mxr` compiled programs |
| `optimised/` | Hardware-optimised compiled programs |

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
