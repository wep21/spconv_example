#pragma once
#include <tensorview/gemm/math/all.h>
#include <tensorview/gemm/core/constants.h>
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T8816_200_C301LLL_SKD_S8 {
namespace out_op {
namespace unaryop {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct UnaryActivation {
  __forceinline__ __device__ tv::array<tv::half_t, 16, 0> operator()(const tv::array<tv::half_t, 16, 0> & src, tv::gemm::Activation type, float alpha, float beta)   {
    
    namespace op = tv::arrayops;
    using scalar_nv_t = tv::equivalent_data_type_t<tv::half_t>;
    auto& src_nv = reinterpret_cast<const tv::array<tv::equivalent_data_type_t<tv::half_t>, 16, 0>&>(src);
    using MathOp = op::MathScalarOp<tv::equivalent_data_type_t<tv::half_t>>;
    switch (type){
        case tv::gemm::Activation::kNone:
            return src;
        case tv::gemm::Activation::kReLU:{
            tv::math::maximum<tv::array<tv::half_t, 16, 0>> max_op;
            return max_op(src, tv::half_t(0));
        }
        case tv::gemm::Activation::kLeakyReLU:{
            tv::array<tv::half_t, 16, 0> res;
            TV_PRAGMA_UNROLL
            for (int i = 0; i < 16; ++i){
                auto x = src[i];
                res[i] = x >= tv::half_t(0) ? x : x * tv::half_t(alpha);
            }
            return res;
        }
        case tv::gemm::Activation::kSigmoid:{
            tv::array<tv::half_t, 16, 0> res;
            TV_PRAGMA_UNROLL
            for (int i = 0; i < 16; ++i){
                auto xx = MathOp::exp(MathOp::neg(src_nv[i]));
                res[i] = tv::half_t(1) / (tv::half_t(1) + *reinterpret_cast<tv::half_t*>( &xx ));
            }
            return res;
        }
        default: return src;
    }
    return src;
  }
};
} // namespace unaryop
} // namespace out_op
} // namespace Turing_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T8816_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib