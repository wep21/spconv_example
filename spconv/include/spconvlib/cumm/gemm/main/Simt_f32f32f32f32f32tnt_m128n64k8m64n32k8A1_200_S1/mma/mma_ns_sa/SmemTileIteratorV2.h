#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1/inpitera/maskiter/PitchLinear.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1 {
namespace mma {
namespace mma_ns_sa {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1::inpitera::maskiter::PitchLinear;
struct SmemTileIteratorV2 {
  char * pointer_;
  __forceinline__ __device__  SmemTileIteratorV2(int stride, float* ptr, int thread_id)   {
    auto thread_offset = ThreadMap::initial_offset(thread_id);
    int offset = (thread_offset[1] / 1) * 132
         + thread_offset[0] * 1;
    pointer_ = reinterpret_cast<char *>(ptr + offset);
    // for transposed input, kThreadAccessShape and kIterationDelta is
    // transposed too.
    // inc_strided_ = stride * 1 * sizeof(float);
    // if (false) {
    //     inc_advance_ = 128 * sizeof(float);
    // } else {
    //     inc_advance_ = 8 * stride * sizeof(float);
    // }
    // tv::printf2_block_once(threadIdx.x, "inc_strided_", inc_strided_, inc_advance_, stride_, offset);
  }
  __forceinline__ __device__ void tile_increment(int num)   {
    pointer_ += 4224 * num;
  }
  __forceinline__ __device__ SmemTileIteratorV2& operator++()   {
    pointer_ +=  4224;
    return *this;
  }
  __forceinline__ __device__ SmemTileIteratorV2& operator--()   {
    pointer_ -=  4224;
    return *this;
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<float, 8, 0>& frag, int32_t pointer_offset)   {
    tv::alignedarray<int, 1, 4> *frag_ptr = reinterpret_cast<tv::alignedarray<int, 1, 4> *>(&frag);
    const char * byte_pointer =
        pointer_ + pointer_offset * sizeof(float);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        tv::alignedarray<int, 1, 4> const *access_ptr =
            reinterpret_cast<tv::alignedarray<int, 1, 4> const *>(byte_pointer);
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 8; ++c) {
            int idx = c + s * 8;
            frag_ptr[idx] =
                access_ptr[c * 16 / 1];
        }
        if (s < 1 - 1) {
            byte_pointer +=  528;
        }
    }
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<float, 8, 0> const & frag, int32_t pointer_offset)   {
    tv::alignedarray<int, 1, 4> const *frag_ptr = reinterpret_cast<tv::alignedarray<int, 1, 4> const *>(&frag);
    char * byte_pointer =
        pointer_ + pointer_offset * sizeof(float);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        tv::alignedarray<int, 1, 4> *access_ptr =
            reinterpret_cast<tv::alignedarray<int, 1, 4> *>(byte_pointer);
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 8; ++c) {
            int idx = c + s * 8;
            access_ptr[c * 16 / 1] =
                frag_ptr[idx];
        }
        if (s < 0) {
            byte_pointer += 528;
        }
    }
  }
  __forceinline__ __device__ void store(tv::array<float, 8, 0> const & frag)   {
    store_with_pointer_offset(frag, 0);
  }
};
} // namespace mma_ns_sa
} // namespace mma
} // namespace Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib