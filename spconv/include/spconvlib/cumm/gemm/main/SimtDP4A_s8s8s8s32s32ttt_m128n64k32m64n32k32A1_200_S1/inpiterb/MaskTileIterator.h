#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/SimtDP4A_s8s8s8s32s32ttt_m128n64k32m64n32k32A1_200_S1/inpiterb/maskiter/PitchLinear.h>
#include <spconvlib/cumm/gemm/main/gpSimtDP4A_s8s8s8s32s32ttt_m128n64k32m64n32k32A1_200_S1/iterb_p/MaskTileIteratorParams.h>
#include <spconvlib/cumm/gemm/main/SimtDP4A_s8s8s8s32s32ttt_m128n64k32m64n32k32A1_200_S1/inpiterb/maskiter/GlobalLoad.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace SimtDP4A_s8s8s8s32s32ttt_m128n64k32m64n32k32A1_200_S1 {
namespace inpiterb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::gemm::main::SimtDP4A_s8s8s8s32s32ttt_m128n64k32m64n32k32A1_200_S1::inpiterb::maskiter::PitchLinear;
using Params = spconvlib::cumm::gemm::main::gpSimtDP4A_s8s8s8s32s32ttt_m128n64k32m64n32k32A1_200_S1::iterb_p::MaskTileIteratorParams;
using GlobalLoad = spconvlib::cumm::gemm::main::SimtDP4A_s8s8s8s32s32ttt_m128n64k32m64n32k32A1_200_S1::inpiterb::maskiter::GlobalLoad;
struct MaskTileIterator {
  const char * pointer_;
  Params const & params_;
  tv::array<int, 2> extent_;
  tv::array<int, 2> thread_offset_;
  int residue_offset_;
  bool is_residue_tile_;
  uint32_t predicates_[1];
  __forceinline__ __device__  MaskTileIterator(Params const & params, const int8_t * ptr, tv::array<int, 2> extent, int thread_id, const tv::array<int, 2>& threadblock_offset) : params_(params), pointer_(reinterpret_cast<const char *>(ptr)), extent_(extent), is_residue_tile_(true)  {
    
    int residue_size = (extent[0] - threadblock_offset[0]) %
                    32;
    if (!residue_size) {
        residue_size = 32;
    }
    // tv::printf2_once(params_.inc_strided_, params_.inc_next_);
        residue_offset_ = 
    (extent[0] - threadblock_offset[0] - 1) /
    32 * 32
    ;
        tv::array<int, 2> residue_extent{
    std::min(threadblock_offset[0] + residue_size, extent_[0])
    , extent_[1]};
    // residue tile always first k axis tile
    // thread_id / kAccessShape[kContig] is 'sub-tile' coord, so we need to
    // convert back to element coord
    thread_offset_ = threadblock_offset + ThreadMap::initial_offset(thread_id);
    // auto init = ThreadMap::initial_offset(thread_id);
    if TV_IF_CONSTEXPR (!false) {
        thread_offset_[0] += residue_offset_;
    }
    // tv::printf2_block_once(threadIdx.x, thread_offset_[0] * extent_[1] + thread_offset_[1]);
    if TV_IF_CONSTEXPR (!false) {
        compute_predicates_(extent, false);
    } else {
        compute_predicates_(residue_extent, false);
    }
    // here we can't use extent_[1] because splitk may split stride.
    add_pointer_offset(thread_offset_[0] * params.stride_ + thread_offset_[1]);
  }
  __forceinline__ __device__ void update_indices()   {
    
  }
  __forceinline__ __device__ void add_pointer_offset(int64_t offset)   {
    pointer_ += sizeof(int8_t) * offset;
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    
    if (is_residue_tile_){
      thread_offset_[0] -= residue_offset_;
      pointer_ -= sizeof(int8_t) * params_.stride_ * residue_offset_;
      compute_predicates_(extent_, true);
      pointer_ += params_.inc_advance_ * (num_tile - 1);
    }
    else{
      pointer_ += params_.inc_advance_ * num_tile;
    }
    is_residue_tile_ = false;
  }
  __forceinline__ __device__ const tv::alignedarray<int, 1, 4> * get(int s, int c, int ss, int v)  const {
    
    return reinterpret_cast<const tv::alignedarray<int, 1, 4> *>(
            pointer_ +
            (ss * params_.stride_ + c * 1) *
                sizeof(int8_t)) +
        v;
  }
  __forceinline__ __device__ void inc_stride()   {
    pointer_ += params_.inc_strided_; 
  }
  __forceinline__ __device__ void end_iter()   {
    pointer_ += params_.inc_next_;
    pointer_ -= params_.inc_advance_;
  }
  __forceinline__ __device__ bool valid(int s, int c, int ss, int v)   {
    
    int scalar_index =
        s * 4 + c * 4 +
        ss * 1 + v * 1;
    int word_idx = scalar_index / 16;
    int residual = scalar_index % 16;
    int byte_idx = residual / 4;
    int bit_idx = residual % 4;
    bool pred = (predicates_[word_idx] & (1u << (byte_idx * 8 + bit_idx))) != 0;
    return pred;
  }
  __forceinline__ __device__ void clear_mask()   {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        predicates_[i] = 0u;
    }
  }
  __forceinline__ __device__ MaskTileIterator & operator++()   {
    tile_increment(1);
    return *this;
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<int8_t, 16, 0>& frag, int32_t pointer_offset)   {
    load_with_byte_offset(frag, pointer_offset * sizeof(int8_t));
  }
  __forceinline__ __device__ void store_with_pointer_offset()   {
    
  }
  __forceinline__ __device__ const tv::alignedarray<int, 1, 4>* load_ptr_with_param(int s, int c, bool& valid_ref)   {
    
    auto ret_ptr = get(s, c, 0, 0);
    valid_ref = valid(s, c, 0, 0);
    if(c == 0){
        if(s != 0)
            inc_stride();
        else
            end_iter();
        }
    return ret_ptr;
  }
  __forceinline__ __device__ void load_with_byte_offset(tv::array<int8_t, 16, 0>& frag, int64_t byte_offset)   {
    
    tv::alignedarray<int, 1, 4> *frag_ptr = reinterpret_cast<tv::alignedarray<int, 1, 4>  *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s){
      TV_PRAGMA_UNROLL
      for (int c = 0; c < 1; ++c){
        TV_PRAGMA_UNROLL
        for (int ss = 0; ss < 4; ++ss){
          TV_PRAGMA_UNROLL
          for (int v = 0; v < 1; ++v){
            int idx =
                s * 4 + c * 4 +
                ss * 1 + v * 1;
            char const *byte_ptr =
                reinterpret_cast<char const *>(get(s, c, ss, v)) + byte_offset;
            tv::alignedarray<int, 1, 4> const *access_ptr =
                reinterpret_cast<tv::alignedarray<int, 1, 4> const *>(byte_ptr);
            tv::gemm::global_load<tv::alignedarray<int, 1, 4>, sizeof(tv::alignedarray<int, 1, 4>)>(
                frag_ptr[idx], access_ptr, valid(s, c, ss, v));
          }
        }
      }
      if (s != 1 - 1) {
          inc_stride();
      }
    }
    end_iter();
    using SubTileShape = tv::mp_list_int<4, 4>;
    tv::gemm::transform::Transpose<16, SubTileShape,
                        int8_t>
        t;
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 1; ++c) {
            int idx = s * 1 + c;
            t.transform(frag.data() + idx * 16, frag.data() + idx * 16);
        }
    }
  }
  __forceinline__ __device__ void store_with_byte_offset()   {
    
  }
  __forceinline__ __device__ void load(tv::array<int8_t, 16, 0>& frag)   {
    load_with_byte_offset(frag, 0);
  }
  __forceinline__ __device__ void store()   {
    
  }
  __forceinline__ __device__ void compute_predicates_(tv::array<int, 2> extent, bool steady = false)   {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        predicates_[i] = 0;
    }
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 1; ++c) {
            TV_PRAGMA_UNROLL
            for (int ss = 0; ss < 4; ++ss) {
                TV_PRAGMA_UNROLL
                for (int v = 0; v < 1; ++v) {
                    tv::array<int, 2> elem_coord{
                        s * 32 + ss + thread_offset_[0],
                        c * 1 + v * 4 +
                            thread_offset_[1]};
                    bool valid;
                    if (steady) {
                        if (0 == 1) {
                            valid = elem_coord[0] < extent[0];
                        } else {
                            valid = elem_coord[1] < extent[1];
                        }
                    } else {
                        valid = elem_coord[0] < extent[0] &&
                                elem_coord[1] < extent[1];
                    }
                    int scalar_index =
                        s * 4 + c * 4 +
                        ss * 1 + v * 1;
                    int word_idx = scalar_index / 16;
                    int residual = scalar_index % 16;
                    int byte_idx = residual / 4;
                    int bit_idx = residual % 4;
                    predicates_[word_idx] |=
                        (unsigned(valid) << (byte_idx * 8 + bit_idx));
                }
            }
        }
    }
  }
};
} // namespace inpiterb
} // namespace SimtDP4A_s8s8s8s32s32ttt_m128n64k32m64n32k32A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib