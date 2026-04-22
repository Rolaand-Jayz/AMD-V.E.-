#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include "ave/backends/migraphx_backend.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "migraphx_backend_tests failed: " << message << '\n';
    std::abort();
}

class ScopedEnvOverride {
  public:
    ScopedEnvOverride(const char* name, std::optional<std::string> value)
        : name_(name) {
        const char* current = std::getenv(name_);
        if (current != nullptr) {
            previous_ = std::string(current);
        }

        if (value.has_value()) {
#if defined(_WIN32)
            _putenv_s(name_, value->c_str());
#else
            setenv(name_, value->c_str(), 1);
#endif
        } else {
#if defined(_WIN32)
            _putenv_s(name_, "");
#else
            unsetenv(name_);
#endif
        }
    }

    ~ScopedEnvOverride() {
        if (previous_.has_value()) {
#if defined(_WIN32)
            _putenv_s(name_, previous_->c_str());
#else
            setenv(name_, previous_->c_str(), 1);
#endif
        } else {
#if defined(_WIN32)
            _putenv_s(name_, "");
#else
            unsetenv(name_);
#endif
        }
    }

  private:
    const char* name_;
    std::optional<std::string> previous_;
};

void testDefaultCompileOptions() {
    ScopedEnvOverride offload("AVE_MIGRAPHX_OFFLOAD_COPY", std::nullopt);
    ScopedEnvOverride precision("AVE_MIGRAPHX_PRECISION", std::nullopt);

    ave::MiGraphXBackend backend;
    const auto opts = backend.compileOptions();
    check(opts.offloadCopy, "default compile options should keep offload_copy enabled");
    check(opts.precision == ave::MiGraphXPrecision::Fp16,
          "default compile precision should remain fp16");
}

void testOffloadCopyEnvOverride() {
    ScopedEnvOverride offload("AVE_MIGRAPHX_OFFLOAD_COPY", std::string("0"));

    ave::MiGraphXBackend backend;
    const auto opts = backend.compileOptions();
#ifdef AVE_HAVE_HIP
    check(!opts.offloadCopy, "AVE_MIGRAPHX_OFFLOAD_COPY=0 should disable implicit offload copy");
    check(opts.format().find("offload_copy=0") != std::string::npos,
          "compile option formatting should expose offload_copy=0");
#else
    check(opts.offloadCopy,
          "AVE_MIGRAPHX_OFFLOAD_COPY=0 should fall back to offload_copy=1 without HIP");
    check(opts.format().find("offload_copy=1") != std::string::npos,
          "compile option formatting should expose non-HIP offload_copy fallback");
#endif

    std::string error;
#ifdef AVE_HAVE_HIP
    check(opts.validate(error),
          "offload_copy=0 should validate on HIP-enabled builds");
    check(error.empty(), "valid offload_copy=0 should not populate an error");
#else
    check(opts.validate(error),
          "offload_copy=0 should validate after non-HIP fallback to offload_copy=1");
    check(error.empty(), "non-HIP offload_copy fallback should not populate an error");
#endif
}

void testInvalidOffloadCopyEnvFallsBack() {
    ScopedEnvOverride offload("AVE_MIGRAPHX_OFFLOAD_COPY", std::string("bogus"));

    ave::MiGraphXBackend backend;
    const auto opts = backend.compileOptions();
    check(opts.offloadCopy,
          "invalid AVE_MIGRAPHX_OFFLOAD_COPY should fall back to offload_copy=1");
}

}  // namespace

int main() {
    testDefaultCompileOptions();
    testOffloadCopyEnvOverride();
    testInvalidOffloadCopyEnvFallsBack();
    return 0;
}
