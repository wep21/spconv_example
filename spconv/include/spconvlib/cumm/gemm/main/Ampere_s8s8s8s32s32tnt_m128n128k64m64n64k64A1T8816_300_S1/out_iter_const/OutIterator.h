#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/gpAmpere_s8s8s8s32s32tnt_m128n128k64m64n64k64A1T8816_300_S1/out_params_ns/tmap/Out5DLinear.h>
#include <spconvlib/cumm/gemm/main/gpAmpere_s8s8s8s32s32tnt_m128n128k64m64n64k64A1T8816_300_S1/out_params_ns/OutIteratorParams.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Ampere_s8s8s8s32s32tnt_m128n128k64m64n64k64A1T8816_300_S1 {
namespace out_iter_const {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::gemm::main::gpAmpere_s8s8s8s32s32tnt_m128n128k64m64n64k64A1T8816_300_S1::out_params_ns::tmap::Out5DLinear;
using Params = spconvlib::cumm::gemm::main::gpAmpere_s8s8s8s32s32tnt_m128n128k64m64n64k64A1T8816_300_S1::out_params_ns::OutIteratorParams;
struct OutIterator {
  const int8_t * pointer_;
  Params const& params_;
  bool column_masks_[2][1];
  int32_t extent_row_;
  int32_t thread_start_row_;
  int counts_[3];
  int64_t indices_[1];
  __forceinline__ __device__  OutIterator(Params const& params, const int8_t * ptr, tv::array<int, 2> extent, tv::array<int, 2> offset_2d, int thread_idx) : params_(params)  {
    // pointer_bkp_ = ptr;
    counts_[0] = 0;
    counts_[1] = 0;
    counts_[2] = 0;
    auto thread_offset = ThreadMap::initial_offset(thread_idx) + offset_2d;
    TV_PRAGMA_UNROLL
    for (int c = 0; c < 2; ++c) {
        for (int v = 0; v < 1; ++v){
            column_masks_[c][v] = ((thread_offset[1] + 64 * c + v * 8) < extent[1]);
        }
    }
    // tv::printf2_block_once("Outthread_offset ", threadIdx.x, thread_offset[0], thread_offset[1], (thread_offset[0]) * stride + thread_offset[1]);
    extent_row_ = extent[0];
    thread_start_row_ = thread_offset[0];
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster){
      TV_PRAGMA_UNROLL
      for (int group = 0; group < 1; ++group){
        TV_PRAGMA_UNROLL
        for (int row = 0; row < 1; ++row){
          int idx = (row +  1 * (group +  1 * cluster));
          int row_offset =
              row * 4 + group * 1 + cluster * 1;
          bool row_guard = ((row_offset + thread_start_row_) < extent_row_ && params_.indice_ptr_ != nullptr);
          indices_[idx] = row_guard ? int64_t(params_.indice_ptr_[row_offset + thread_start_row_]) * int64_t(params_.stride) : 0;
        }
      }
    }
    pointer_ = ptr + thread_offset[1];
  }
  __forceinline__ __device__ void store_with_offset(tv::array<int8_t, 16, 0> const & frag, int32_t offset)   {
    
  }
  __forceinline__ __device__ void load_with_offset(tv::array<int8_t, 16, 0>  & frag, int32_t offset)   {
    
    auto cur_pointer = pointer_;
    tv::alignedarray<int2, 1, 8>  *frag_ptr = reinterpret_cast<tv::alignedarray<int2, 1, 8>  *>(&frag);
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster){
      TV_PRAGMA_UNROLL
      for (int group = 0; group < 1; ++group){
        TV_PRAGMA_UNROLL
        for (int row = 0; row < 1; ++row){
          int frag_row_idx =
              (row +  1 * (group +  1 * cluster));
          // delta: [Cluster, Group, Row]
          int row_offset =
              row * 4 + group * 1 + cluster * 1;
          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);
          tv::alignedarray<int2, 1, 8> const *memory_pointer =
              reinterpret_cast<tv::alignedarray<int2, 1, 8> const *>(cur_pointer + offset + indices_[frag_row_idx]);
          TV_PRAGMA_UNROLL
          for (int column = 0; column < 2; ++column){
            bool guard = row_guard && column_masks_[column][0];
            tv::gemm::global_load<tv::alignedarray<int2, 1, 8>, sizeof(tv::alignedarray<int2, 1, 8>)>(
                frag_ptr[frag_row_idx *  2 + column],
                (const void *)&memory_pointer[column * 64 / 8],
                guard);
          }
        }
      }
    }
  }
  __forceinline__ __device__ void store(tv::array<int8_t, 16, 0> const & frag)   {
    store_with_offset(frag, 0);
  }
  __forceinline__ __device__ void load(tv::array<int8_t, 16, 0> & frag)   {
    load_with_offset(frag, 0);
  }
  __forceinline__ __device__ OutIterator& operator++()   {
    ++counts_[2];
    // kPartShape: [Tile, Cluster, Group, Row, Col]
    thread_start_row_ += 8;
    if (counts_[2] == 8) {
    counts_[2] = 0;
    ++counts_[1];
    thread_start_row_ +=
        (2 - 1) * 8 * 8;
    if (counts_[1] == 1) {
        counts_[1] = 0;
        ++counts_[0];
        thread_start_row_ +=
            1 * 2 * 8 * 8;
        if (counts_[0] == 1) {
        counts_[0] = 0;
        }
    }
    }
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster){
      TV_PRAGMA_UNROLL
      for (int group = 0; group < 1; ++group){
        TV_PRAGMA_UNROLL
        for (int row = 0; row < 1; ++row){
          int idx =
              (row +  1 * (group +  1 * cluster));
          int row_offset =
              row * 4 + group * 1 + cluster * 1;
          bool row_guard = ((row_offset + thread_start_row_) < extent_row_ && params_.indice_ptr_ != nullptr);
          indices_[idx] = row_guard ? int64_t(params_.indice_ptr_[row_offset + thread_start_row_]) * int64_t(params_.stride) : 0;
        }
      }
    }
    return *this;
  }
};
} // namespace out_iter_const
} // namespace Ampere_s8s8s8s32s32tnt_m128n128k64m64n64k64A1T8816_300_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib