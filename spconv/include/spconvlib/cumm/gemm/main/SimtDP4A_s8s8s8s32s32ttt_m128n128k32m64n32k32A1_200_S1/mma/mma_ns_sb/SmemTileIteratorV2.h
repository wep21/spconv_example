#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/SimtDP4A_s8s8s8s32s32ttt_m128n128k32m64n32k32A1_200_S1/inpiterb/maskiter/PitchLinear.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace SimtDP4A_s8s8s8s32s32ttt_m128n128k32m64n32k32A1_200_S1 {
namespace mma {
namespace mma_ns_sb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::gemm::main::SimtDP4A_s8s8s8s32s32ttt_m128n128k32m64n32k32A1_200_S1::inpiterb::maskiter::PitchLinear;
struct SmemTileIteratorV2 {
  char * pointer_;
  __forceinline__ __device__  SmemTileIteratorV2(int stride, int8_t* ptr, int thread_id)   {
    auto thread_offset = ThreadMap::initial_offset(thread_id);
    int offset = (thread_offset[0] / 4) * 512
        + thread_offset[1] * 4;
    pointer_ = reinterpret_cast<char *>(ptr + offset);
    // for transposed input, kThreadAccessShape and kIterationDelta is
    // transposed too.
    // inc_strided_ = stride * 8 * sizeof(int8_t);
    // if (false) {
    //     inc_advance_ = 128 * sizeof(int8_t);
    // } else {
    //     inc_advance_ = 32 * stride * sizeof(int8_t);
    // }
    // tv::printf2_block_once(threadIdx.x, "inc_strided_", inc_strided_, inc_advance_, stride_, offset);
  }
  __forceinline__ __device__ void tile_increment(int num)   {
    pointer_ += 4096 * num;
  }
  __forceinline__ __device__ SmemTileIteratorV2& operator++()   {
    pointer_ +=  4096;
    return *this;
  }
  __forceinline__ __device__ SmemTileIteratorV2& operator--()   {
    pointer_ -=  4096;
    return *this;
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<int8_t, 16, 0>& frag, int32_t pointer_offset)   {
    tv::alignedarray<int4, 1, 16> *frag_ptr = reinterpret_cast<tv::alignedarray<int4, 1, 16> *>(&frag);
    const char * byte_pointer =
        pointer_ + pointer_offset * sizeof(int8_t);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        tv::alignedarray<int4, 1, 16> const *access_ptr =
            reinterpret_cast<tv::alignedarray<int4, 1, 16> const *>(byte_pointer);
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 1; ++c) {
            int idx = c + s * 1;
            frag_ptr[idx] =
                access_ptr[c * 4 / 16];
        }
        if (s < 1 - 1) {
            byte_pointer +=  4096;
        }
    }
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<int8_t, 16, 0> const & frag, int32_t pointer_offset)   {
    tv::alignedarray<int4, 1, 16> const *frag_ptr = reinterpret_cast<tv::alignedarray<int4, 1, 16> const *>(&frag);
    char * byte_pointer =
        pointer_ + pointer_offset * sizeof(int8_t);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        tv::alignedarray<int4, 1, 16> *access_ptr =
            reinterpret_cast<tv::alignedarray<int4, 1, 16> *>(byte_pointer);
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 1; ++c) {
            int idx = c + s * 1;
            access_ptr[c * 4 / 16] =
                frag_ptr[idx];
        }
        if (s < 0) {
            byte_pointer += 4096;
        }
    }
  }
  __forceinline__ __device__ void store(tv::array<int8_t, 16, 0> const & frag)   {
    store_with_pointer_offset(frag, 0);
  }
};
} // namespace mma_ns_sb
} // namespace mma
} // namespace SimtDP4A_s8s8s8s32s32ttt_m128n128k32m64n32k32A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib