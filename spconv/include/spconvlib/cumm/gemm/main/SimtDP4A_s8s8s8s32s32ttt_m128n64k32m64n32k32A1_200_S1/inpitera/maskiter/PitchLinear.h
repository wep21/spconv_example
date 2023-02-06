#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace SimtDP4A_s8s8s8s32s32ttt_m128n64k32m64n32k32A1_200_S1 {
namespace inpitera {
namespace maskiter {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct PitchLinear {
  __forceinline__ __host__ __device__ static tv::array<int, 2> initial_offset(int thread_id)   {
    return {(thread_id / 8) * 4,
            (thread_id % 8) *  4};
  }
};
} // namespace maskiter
} // namespace inpitera
} // namespace SimtDP4A_s8s8s8s32s32ttt_m128n64k32m64n32k32A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib