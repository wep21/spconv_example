#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/gemm/main/Turing_s8s8s8s32s32tnt_m128n64k32m64n32k32A1T8816_200_S1/out_op/LinearCombination.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_s8s8s8s32s32tnt_m128n64k32m64n32k32A1T8816_200_S1 {
namespace output {
namespace out_ns_apply {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using OutputOp = spconvlib::cumm::gemm::main::Turing_s8s8s8s32s32tnt_m128n64k32m64n32k32A1T8816_200_S1::out_op::LinearCombination;
struct ApplyOutputOp {
  __forceinline__ __device__ static void apply_output_operator(tv::array<int8_t, 8, 0> & output_fragment, OutputOp const & output_op, tv::array<int32_t, 8, 0> const & aligned_accum_fragment, tv::array<int8_t, 8, 0> const & source_fragment)   {
    
    constexpr int kOutFragCount = tv::array_size_v<tv::array<int8_t, 8, 0>>;
    using OutAccessType = tv::array<typename tv::array<int8_t, 8, 0>::value_type, 8, 0>;
    using InputAccessType = tv::array<typename tv::array<int32_t, 8, 0>::value_type, 8, 0>;
    OutAccessType *output_frag_ptr =
        reinterpret_cast<OutAccessType *>(&output_fragment);
    InputAccessType const *compute_frag_ptr =
        reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
    OutAccessType const *source_frag_ptr =
        reinterpret_cast<OutAccessType const *>(&source_fragment);
    constexpr int kOutOpIterations = kOutFragCount / 8;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < kOutOpIterations; ++i) {
        output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
    }
  }
  __forceinline__ __device__ static void apply_output_operator_no_source(tv::array<int8_t, 8, 0> & output_fragment, OutputOp const & output_op, tv::array<int32_t, 8, 0> const & aligned_accum_fragment)   {
    
    constexpr int kOutFragCount = tv::array_size_v<tv::array<int8_t, 8, 0>>;
    using OutAccessType = tv::array<typename tv::array<int8_t, 8, 0>::value_type, 8, 0>;
    using InputAccessType = tv::array<typename tv::array<int32_t, 8, 0>::value_type, 8, 0>;
    OutAccessType *output_frag_ptr =
        reinterpret_cast<OutAccessType *>(&output_fragment);
    InputAccessType const *compute_frag_ptr =
        reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
    constexpr int kOutOpIterations = kOutFragCount / 8;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < kOutOpIterations; ++i) {
        output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
    }
  }
};
} // namespace out_ns_apply
} // namespace output
} // namespace Turing_s8s8s8s32s32tnt_m128n64k32m64n32k32A1T8816_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib