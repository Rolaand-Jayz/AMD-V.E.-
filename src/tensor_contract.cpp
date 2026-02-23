// ─────────────────────────────────────────────────────────────────
// tensor_contract.cpp — TensorShape and TensorContract implementation
// ─────────────────────────────────────────────────────────────────
#include "ave/tensor_contract.hpp"

#include <numeric>
#include <sstream>

namespace ave {

// ─────────────────────────────────────────────────────────────────
// toString
// ─────────────────────────────────────────────────────────────────
std::string toString(TensorLayout layout) {
    switch (layout) {
        case TensorLayout::NCHW:    return "NCHW";
        case TensorLayout::NHWC:    return "NHWC";
        case TensorLayout::CHW:     return "CHW";
        case TensorLayout::HWC:     return "HWC";
        case TensorLayout::Unknown: return "Unknown";
    }
    return "Unknown";
}

std::string toString(TensorDtype dtype) {
    switch (dtype) {
        case TensorDtype::Fp32:          return "fp32";
        case TensorDtype::Fp16:          return "fp16";
        case TensorDtype::Bf16:          return "bf16";
        case TensorDtype::Int8:          return "int8";
        case TensorDtype::Fp8E4M3FNUZ:   return "fp8e4m3fnuz";
        case TensorDtype::Unknown:       return "Unknown";
    }
    return "Unknown";
}

// ─────────────────────────────────────────────────────────────────
// TensorShape
// ─────────────────────────────────────────────────────────────────
std::int64_t TensorShape::elements() const {
    if (dims.empty()) { return 0; }
    return std::accumulate(dims.begin(), dims.end(),
                           std::int64_t{1},
                           std::multiplies<std::int64_t>{});
}

std::string TensorShape::format() const {
    if (dims.empty()) { return "<empty>"; }
    std::ostringstream os;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) { os << '\xc3' << '\x97'; }  // UTF-8 '×'
        os << dims[i];
    }
    return os.str();
}

bool TensorShape::operator==(const TensorShape& other) const {
    return dims == other.dims;
}
bool TensorShape::operator!=(const TensorShape& other) const {
    return !(*this == other);
}

// ─────────────────────────────────────────────────────────────────
// TensorContract
// ─────────────────────────────────────────────────────────────────
bool TensorContract::validate(std::string& error) const {
    if (name.empty()) {
        error = "TensorContract: parameter name must not be empty.";
        return false;
    }
    if (shape.dims.empty()) {
        error = "TensorContract '" + name + "': shape.dims must not be empty.";
        return false;
    }
    for (const auto d : shape.dims) {
        if (d <= 0) {
            error = "TensorContract '" + name
                    + "': dimension must be positive (got "
                    + std::to_string(d) + ").";
            return false;
        }
    }
    if (layout == TensorLayout::Unknown) {
        error = "TensorContract '" + name
                + "': layout must not be Unknown in a production contract.";
        return false;
    }
    if (dtype == TensorDtype::Unknown) {
        error = "TensorContract '" + name
                + "': dtype must not be Unknown in a production contract.";
        return false;
    }
    return true;
}

std::string TensorContract::format() const {
    return name + " [" + shape.format() + " "
           + toString(layout) + " " + toString(dtype) + "]";
}

// ─────────────────────────────────────────────────────────────────
// Assertion helpers
// ─────────────────────────────────────────────────────────────────
bool assertElementCount(const TensorContract& contract,
                        std::size_t           actualElements,
                        std::string&          error) {
    const auto expected = static_cast<std::size_t>(contract.shape.elements());
    if (actualElements != expected) {
        error = "Tensor contract violation for '" + contract.name
                + "': expected " + std::to_string(expected)
                + " elements, got " + std::to_string(actualElements)
                + " (" + contract.shape.format() + " "
                + toString(contract.dtype) + ").";
        return false;
    }
    return true;
}

bool assertContractsMatch(const TensorContract& expected,
                          const TensorContract& actual,
                          std::string&          error) {
    if (expected.shape != actual.shape) {
        error = "Tensor shape mismatch: expected "
                + expected.format() + " but got " + actual.format() + ".";
        return false;
    }
    if (expected.dtype != actual.dtype) {
        error = "Tensor dtype mismatch: expected "
                + toString(expected.dtype) + " but got "
                + toString(actual.dtype) + " (contract: '"
                + expected.name + "').";
        return false;
    }
    return true;
}

}  // namespace ave
