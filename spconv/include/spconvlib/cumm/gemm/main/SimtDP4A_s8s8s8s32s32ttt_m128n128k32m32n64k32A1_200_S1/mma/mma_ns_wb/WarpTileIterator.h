#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/SimtDP4A_s8s8s8s32s32ttt_m128n128k32m32n64k32A1_200_S1/mma/mma_ns_wb/ns1/RowMajorInterleaved.h>
#include <spconvlib/cumm/gemm/main/SimtDP4A_s8s8s8s32s32ttt_m128n128k32m32n64k32A1_200_S1/mma/mma_ns_wa/ns2/RowMajorInterleaved.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace SimtDP4A_s8s8s8s32s32ttt_m128n128k32m32n64k32A1_200_S1 {
namespace mma {
namespace mma_ns_wb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using SmemLayout = spconvlib::cumm::gemm::main::SimtDP4A_s8s8s8s32s32ttt_m128n128k32m32n64k32A1_200_S1::mma::mma_ns_wb::ns1::RowMajorInterleaved;
using LaneLayout = spconvlib::cumm::gemm::main::SimtDP4A_s8s8s8s32s32ttt_m128n128k32m32n64k32A1_200_S1::mma::mma_ns_wa::ns2::RowMajorInterleaved;
struct WarpTileIterator {
  tv::array<int8_t, 16> * pointer_;
  __forceinline__ __device__  WarpTileIterator(int8_t * ptr, int warp_idx_k, int warp_idx_residual, int lane_idx)   {
    constexpr auto lane_layout = LaneLayout::from_shape({4, 8});
    constexpr auto smem_layout = SmemLayout::from_shape({32, 128});
    tv::array<int, 2> logical_offset{warp_idx_k * 8,
                                    warp_idx_residual * 64};
    // int lane_offset;
    if (false) {
        // 0, 1, 0, 1, 0, 1, ..., 2, 3, 2, 3, 2, 3
        logical_offset[1] += lane_layout.inverse_0(lane_idx) * 4;
    } else {
        // 0, 0, 1, 1, 2, 2, 3, 3, ...
        logical_offset[1] += lane_layout.inverse_1(lane_idx) * 4;
    }
    auto offset = smem_layout(logical_offset[0], logical_offset[1]);
    pointer_ = reinterpret_cast<tv::array<int8_t, 16> *>(ptr + offset);
  }
  __forceinline__ __device__ void tile_increment(int num)   {
    static constexpr auto smem_layout = SmemLayout::from_shape({32, 128});
    pointer_ += smem_layout(num * 4, 0) / 16;
  }
  __forceinline__ __device__ WarpTileIterator & operator++()   {
    static constexpr auto smem_layout = SmemLayout::from_shape({32, 128});
    auto offset = 
    pointer_ += smem_layout(4, 0) / 16;
    return *this;
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<int8_t, 32, 0>& frag, int32_t pointer_offset)   {
    static constexpr auto smem_layout = SmemLayout::from_shape({32, 128});
    tv::array<int8_t, 16> * dst_ptr = reinterpret_cast<tv::array<int8_t, 16> *>(&frag);
    // kRow: 1, kCol: 2
    TV_PRAGMA_UNROLL
    for (int k = 0; k < 1; ++k) {
        TV_PRAGMA_UNROLL
        for (int n = 0; n < 2; ++n) {
            // ref offset: [8, 128 / 4], (0, n * 8) ~= (0, 32)
            // lane_id = 0, [0, 0-3], [0, 32-35]
            // lane_id = 1, [0, 0-3], [0, 32-35]
            // lane_id = 2, [0, 4-7], [0, 36-39]
            // lane_id = 3, [0, 4-7], [0, 36-39]
            // ...
            // lane_id = 31, [0, 28-31], [0, 60-63]
            auto offset = smem_layout(k * 4,
                                    n * (false ? 4 : 8) *
                                        4) /
                        16;
            // auto offset = smem_layout(k * 4,
            //                         n * 2);
            TV_PRAGMA_UNROLL
            for (int sub = 0; sub < 1; ++sub){
                dst_ptr[k * 2 
                    + n * 1 + sub] =
                    pointer_[offset + sub + pointer_offset / 16];
            }
        }
    }
  }
  __forceinline__ __device__ void load(tv::array<int8_t, 32, 0>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void set_kgroup_index(int wmma_k)   {
    
  }
};
} // namespace mma_ns_wb
} // namespace mma
} // namespace SimtDP4A_s8s8s8s32s32ttt_m128n128k32m32n64k32A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib