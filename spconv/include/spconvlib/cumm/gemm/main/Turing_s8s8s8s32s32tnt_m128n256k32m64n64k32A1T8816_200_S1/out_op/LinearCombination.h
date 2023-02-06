#pragma once
#include <tensorview/gemm/math/all.h>
#include <tensorview/gemm/core/constants.h>
#include <spconvlib/cumm/gemm/main/Turing_s8s8s8s32s32tnt_m128n256k32m64n64k32A1T8816_200_S1/out_op/unaryop/Clamp.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_s8s8s8s32s32tnt_m128n256k32m64n64k32A1T8816_200_S1 {
namespace out_op {
using UnaryOp = spconvlib::cumm::gemm::main::Turing_s8s8s8s32s32tnt_m128n256k32m64n64k32A1T8816_200_S1::out_op::unaryop::Clamp;
struct LinearCombination {
  int32_t alpha;
  int32_t beta;
  int32_t act_alpha;
  int32_t act_beta;
  tv::gemm::Activation act_type;
  __forceinline__ __device__  LinearCombination(int32_t alpha = int32_t(1), int32_t beta = int32_t(0), int32_t act_alpha = int32_t(0), int32_t act_beta = int32_t(0), tv::gemm::Activation type = tv::gemm::Activation::kNone) : alpha(alpha), beta(beta), act_alpha(act_alpha), act_beta(act_beta), act_type(type)  {
    
  }
  __forceinline__ __device__ bool is_source_needed()  const {
    return  beta != int32_t(0);
  }
  __forceinline__ __device__ void set_k_partition(int k_part, int k_part_count)   {
    
    if (k_part) {
        beta = int32_t(1);
    }
  }
  __forceinline__ __device__ tv::array<int8_t, 16, 0> operator()(tv::array<int32_t, 16, 0> const& accumulator, tv::array<int8_t, 16, 0> const& source)  const {
    
    tv::gemm::NumericArrayConverter<int32_t, int8_t, 16, tv::gemm::FloatRoundStyle::round_to_nearest>
        source_converter;
    tv::gemm::NumericArrayConverter<int32_t, int32_t, 16, tv::gemm::FloatRoundStyle::round_to_nearest>
        accumulator_converter;
    tv::array<int32_t, 16, 0> converted_source = source_converter(source);
    tv::array<int32_t, 16, 0> converted_accumulator = accumulator_converter(accumulator);
    tv::array<int32_t, 16, 0> intermediate;
    tv::math::multiplies<tv::array<int32_t, 16, 0>> mul_add_source;
    tv::math::multiply_add<tv::array<int32_t, 16, 0>> mul_add_accumulator;
    intermediate =
        mul_add_source(beta, converted_source); // X =  beta * C + uniform
    intermediate = mul_add_accumulator(alpha, converted_accumulator,
                                    intermediate); // D = alpha * Accum + X
    UnaryOp op;
    intermediate = op(intermediate, act_type, act_alpha, act_beta);
    // Convert to destination numeric type
    tv::gemm::NumericArrayConverter<int8_t, int32_t, 16, tv::gemm::FloatRoundStyle::round_to_nearest>
        destination_converter;
    return destination_converter(intermediate);
  }
  __forceinline__ __device__ tv::array<int8_t, 16, 0> operator()(tv::array<int32_t, 16, 0> const& accumulator)  const {
    
    tv::gemm::NumericArrayConverter<int32_t, int32_t, 16, tv::gemm::FloatRoundStyle::round_to_nearest>
        accumulator_converter;
    tv::array<int32_t, 16, 0> converted_accumulator = accumulator_converter(accumulator);
    tv::array<int32_t, 16, 0> intermediate;
    tv::math::multiplies<tv::array<int32_t, 16, 0>> mul_accumulator;
    intermediate = mul_accumulator(alpha, converted_accumulator); // D = alpha * Accum + X
    UnaryOp op;
    intermediate = op(intermediate, act_type, act_alpha, act_beta);
    // Convert to destination numeric type
    tv::gemm::NumericArrayConverter<int8_t, int32_t, 16, tv::gemm::FloatRoundStyle::round_to_nearest>
        destination_converter;
    return destination_converter(intermediate);
  }
};
} // namespace out_op
} // namespace Turing_s8s8s8s32s32tnt_m128n256k32m64n64k32A1T8816_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib