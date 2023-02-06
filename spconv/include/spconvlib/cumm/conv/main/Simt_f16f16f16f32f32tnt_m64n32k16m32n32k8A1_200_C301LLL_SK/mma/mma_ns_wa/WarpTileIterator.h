#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Simt_f16f16f16f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK/mma/mma_ns_wa/ns1/RowMajorInterleaved.h>
#include <spconvlib/cumm/conv/main/Simt_f16f16f16f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK/mma/mma_ns_wa/ns2/RowMajorInterleaved.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f16f16f16f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK {
namespace mma {
namespace mma_ns_wa {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using SmemLayout = spconvlib::cumm::conv::main::Simt_f16f16f16f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK::mma::mma_ns_wa::ns1::RowMajorInterleaved;
using LaneLayout = spconvlib::cumm::conv::main::Simt_f16f16f16f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK::mma::mma_ns_wa::ns2::RowMajorInterleaved;
struct WarpTileIterator {
  tv::array<tv::half_t, 4> * pointer_;
  int wmma_k_index_;
  __forceinline__ __device__  WarpTileIterator(tv::half_t * ptr, int warp_idx_k, int warp_idx_residual, int lane_idx) : wmma_k_index_(0)  {
    constexpr auto lane_layout = LaneLayout::from_shape({4, 8});
    constexpr auto smem_layout = SmemLayout::from_shape({16, 68});
    tv::array<int, 2> logical_offset{warp_idx_k * 8,
                                    warp_idx_residual * 32};
    // int lane_offset;
    if (true) {
        // 0, 1, 0, 1, 0, 1, ..., 2, 3, 2, 3, 2, 3
        logical_offset[1] += lane_layout.inverse_0(lane_idx) * 8;
    } else {
        // 0, 0, 1, 1, 2, 2, 3, 3, ...
        logical_offset[1] += lane_layout.inverse_1(lane_idx) * 8;
    }
    auto offset = smem_layout(logical_offset[0], logical_offset[1]);
    pointer_ = reinterpret_cast<tv::array<tv::half_t, 4> *>(ptr + offset);
  }
  __forceinline__ __device__ void tile_increment(int num)   {
    static constexpr auto smem_layout = SmemLayout::from_shape({16, 68});
    pointer_ += smem_layout(num * 1, 0) / 4;
  }
  __forceinline__ __device__ WarpTileIterator & operator++()   {
    static constexpr auto smem_layout = SmemLayout::from_shape({16, 68});
    auto offset = 
    pointer_ += smem_layout(1, 0) / 4;
    ++wmma_k_index_;
    if (wmma_k_index_ == 8){
        wmma_k_index_ = 0;
        pointer_ += smem_layout(8, 0) / 4;
        // tile_increment(1);
    }
    return *this;
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<tv::half_t, 8, 0>& frag, int32_t pointer_offset)   {
    static constexpr auto smem_layout = SmemLayout::from_shape({16, 68});
    tv::array<tv::half_t, 4> * dst_ptr = reinterpret_cast<tv::array<tv::half_t, 4> *>(&frag);
    // kRow: 1, kCol: 2
    TV_PRAGMA_UNROLL
    for (int k = 0; k < 1; ++k) {
        TV_PRAGMA_UNROLL
        for (int n = 0; n < 1; ++n) {
            // ref offset: [8, 128 / 4], (0, n * 8) ~= (0, 32)
            // lane_id = 0, [0, 0-3], [0, 32-35]
            // lane_id = 1, [0, 0-3], [0, 32-35]
            // lane_id = 2, [0, 4-7], [0, 36-39]
            // lane_id = 3, [0, 4-7], [0, 36-39]
            // ...
            // lane_id = 31, [0, 28-31], [0, 60-63]
            auto offset = smem_layout(k * 1,
                                    n * (true ? 4 : 8) *
                                        8) /
                        4;
            // auto offset = smem_layout(k * 1,
            //                         n * 8);
            TV_PRAGMA_UNROLL
            for (int sub = 0; sub < 2; ++sub){
                dst_ptr[k * 2 
                    + n * 2 + sub] =
                    pointer_[offset + sub + pointer_offset / 4];
            }
        }
    }
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 8, 0>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void set_kgroup_index(int wmma_k)   {
    
    wmma_k_index_ = wmma_k;
  }
};
} // namespace mma_ns_wa
} // namespace mma
} // namespace Simt_f16f16f16f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib