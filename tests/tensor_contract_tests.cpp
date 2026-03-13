#include <cstdlib>
#include <iostream>
#include <vector>

#include "ave/tensor_contract.hpp"

namespace {

using ave::TensorLayout;

void check(bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "tensor_contract_tests failed: " << message << '\n';
    std::abort();
}

void testInferFourDimensionalLayouts() {
    check(ave::inferTensorLayout({1, 3, 720, 1280}) == TensorLayout::NCHW,
          "expected NCHW for channel-first 4D tensor");
    check(ave::inferTensorLayout({1, 720, 1280, 3}) == TensorLayout::NHWC,
          "expected NHWC for channel-last 4D tensor");
    check(ave::inferTensorLayout({1, 32, 720, 1280}) == TensorLayout::NCHW,
          "ambiguous 4D tensor should default to NCHW");
}

void testInferThreeDimensionalLayouts() {
    check(ave::inferTensorLayout({3, 720, 1280}) == TensorLayout::CHW,
          "expected CHW for channel-first 3D tensor");
    check(ave::inferTensorLayout({720, 1280, 3}) == TensorLayout::HWC,
          "expected HWC for channel-last 3D tensor");
}

void testInferUnknownLayoutForNonImageRanks() {
    check(ave::inferTensorLayout({1, 128}) == TensorLayout::Unknown,
          "rank-2 tensor should not infer an image layout");
    check(ave::inferTensorLayout({}) == TensorLayout::Unknown,
          "empty tensor should not infer a layout");
}

}  // namespace

int main() {
    testInferFourDimensionalLayouts();
    testInferThreeDimensionalLayouts();
    testInferUnknownLayoutForNonImageRanks();
    return 0;
}
