#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/layout/RowMajor.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_f16f16f16f32f32tnt_m32n64k32m32n32k16A1T1688_200_C301LLL_SK {
namespace output {
namespace out_ns_warp {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using RowMajor = spconvlib::cumm::gemm::layout::RowMajor;
struct OutWarpTileIteratorTensorOpMixed {
  tv::alignedarray<int2, 1, 8> * pointers_[1];
  RowMajor layout_;
  int warp_column_;
  __forceinline__ __device__  OutWarpTileIteratorTensorOpMixed(float * ptr, int warp_offset_m, int warp_offset_n, int lane_idx) : layout_(36), warp_column_(0)  {
    
    int tensorop_row = lane_idx / 4;
    int tensorop_col = lane_idx % 4;
    auto pointer = reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(ptr) + tensorop_row * 36;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i){
      // int swizzled_tensorop_col = (tensorop_col % 2) + ((
      //     (tensorop_col / 2) + i) % 1) * 2
      int swizzled_tensorop_col = tensorop_col ^ (i * 2);
      pointers_[i] = pointer + swizzled_tensorop_col;
    }
    add_warp_offset(warp_offset_m, warp_offset_n);
  }
  __forceinline__ __device__ void add_warp_offset(int warp_m, int warp_n)   {
    
    auto offset_0 = layout_(
        warp_m * 8,
        warp_n * 32 / 2);
    pointers_[0] += offset_0;
    // tv::printf2_once<' ', 234>("OFFSET", 0, offset_0);
    warp_column_ += warp_n * 32;
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<float, 8, 0> const & frag, int32_t pointer_offset)   {
    const tv::alignedarray<int2, 1, 8> * frag_ptr = reinterpret_cast<const tv::alignedarray<int2, 1, 8> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int n = 0; n < 4; ++n){
      int column_idx = warp_column_ + n * 8;
      int smem_line_offset = column_idx * 4 / 128;
      int ptr_idx = smem_line_offset % 1;
      auto ptr = pointers_[ptr_idx];
      int offset = n * 4 + pointer_offset / 2;
      ptr[offset] = frag_ptr[n];
    }
  }
  __forceinline__ __device__ void store(tv::array<float, 8, 0> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    
    pointers_[0] += pointer_offset / 2;
  }
};
} // namespace out_ns_warp
} // namespace output
} // namespace Turing_f16f16f16f32f32tnt_m32n64k32m32n32k16A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib