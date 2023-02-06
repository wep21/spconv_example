#pragma once
#include <tensorview/gemm/math/all.h>
#include <tensorview/gemm/core/constants.h>
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_s8s8s8s32s32tnt_m128n256k32m64n64k32A1T8816_200_S1 {
namespace out_op {
namespace unaryop {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct Clamp {
  __forceinline__ __device__ tv::array<int32_t, 16, 0> operator()(const tv::array<int32_t, 16, 0> & src, tv::gemm::Activation type, float alpha, float beta)   {
    
    constexpr int32_t kClamp = int32_t((1U << (sizeof(int8_t) * 8 - 1)) - 1);
    tv::math::minimum<tv::array<int32_t, 16, 0>> min_op;
    tv::math::maximum<tv::array<int32_t, 16, 0>> max_op;
    tv::array<int32_t, 16, 0> intermediate = max_op(src, -kClamp - int32_t(1));
    intermediate = min_op(intermediate, kClamp);
    return intermediate;
  }
};
} // namespace unaryop
} // namespace out_op
} // namespace Turing_s8s8s8s32s32tnt_m128n256k32m64n64k32A1T8816_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib