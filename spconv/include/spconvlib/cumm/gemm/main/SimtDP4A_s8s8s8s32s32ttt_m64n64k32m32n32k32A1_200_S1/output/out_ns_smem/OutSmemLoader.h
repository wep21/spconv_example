#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/SimtDP4A_s8s8s8s32s32ttt_m64n64k32m32n32k32A1_200_S1/output/out_ns_smem/tmap/Out5DLinear.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace SimtDP4A_s8s8s8s32s32ttt_m64n64k32m32n32k32A1_200_S1 {
namespace output {
namespace out_ns_smem {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::gemm::main::SimtDP4A_s8s8s8s32s32ttt_m64n64k32m32n32k32A1_200_S1::output::out_ns_smem::tmap::Out5DLinear;
struct OutSmemLoader {
  int32_t * pointer_;
  __forceinline__ __device__  OutSmemLoader(int32_t * ptr, int thread_idx)   {
    auto thread_offset = ThreadMap::initial_offset(thread_idx);
    pointer_ = ptr + thread_offset[0] * 81 + thread_offset[1];
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<int32_t, 4, 0> & frag, int32_t pointer_offset)   {
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster) {
        TV_PRAGMA_UNROLL
        for (int group = 0; group < 2; ++group) {
            TV_PRAGMA_UNROLL
            for (int row = 0; row < 1; ++row) {
                const int32_t * cur_pointer =
                    pointer_ + row * 1 * 81 + group * 2 * 81 +
                    cluster * 1 * 81 + pointer_offset;
                int frag_row_idx =
                    (row + 1 * (group + 2 * cluster));
                tv::alignedarray<int, 1, 4> *frag_ptr = reinterpret_cast<tv::alignedarray<int, 1, 4> *>(&frag);
                tv::alignedarray<int, 1, 4> const *memory_pointer =
                    reinterpret_cast<tv::alignedarray<int, 1, 4> const *>(cur_pointer);
                TV_PRAGMA_UNROLL
                for (int column = 0; column < 2; ++column) {
                    int frag_idx = frag_row_idx * 2 + column;
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < 1; ++v) {
                        frag_ptr[frag_idx * 1 + v] =
                            memory_pointer[(column * 32 / 1) *
                                                1 +
                                            v];
                    }
                }
            }
        }
    }
  }
  __forceinline__ __device__ void load(tv::array<int32_t, 4, 0> & frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    pointer_ += pointer_offset;
  }
};
} // namespace out_ns_smem
} // namespace output
} // namespace SimtDP4A_s8s8s8s32s32ttt_m64n64k32m32n32k32A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib