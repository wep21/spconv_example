#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_s8s8s8s32s32tnt_m64n128k32m32n64k32A1T8816_200_S1 {
namespace inpitera {
namespace maskiter {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct PitchLinearWarpRaked {
  __forceinline__ __host__ __device__ static bool is_skipped(int thread_id)   {
    
    if(!0)
        return false;
    return thread_id >= 128;
  }
  __forceinline__ __host__ __device__ static tv::array<int, 2> initial_offset(int thread_id)   {
    if(is_skipped(thread_id))
        return {0, 0};        // to InputIter: inefficient but convenience dummy offset.
    int warp_id = (thread_id / 32);
    int lane_id = (thread_id % 32);
    tv::array<int, 2> warp_offset{warp_id / 1,
                                warp_id % 1};
    constexpr tv::array<int, 2> kWarpDilation{16, 2};
    tv::array<int, 2> thread_offset_in_warp{lane_id / 2,
                                            lane_id % 2};
    tv::array<int, 2> offset_in_tile =
        kWarpDilation * warp_offset + thread_offset_in_warp;
    return {offset_in_tile[0], offset_in_tile[1] * 16};
  }
};
} // namespace maskiter
} // namespace inpitera
} // namespace Turing_s8s8s8s32s32tnt_m64n128k32m32n64k32A1T8816_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib