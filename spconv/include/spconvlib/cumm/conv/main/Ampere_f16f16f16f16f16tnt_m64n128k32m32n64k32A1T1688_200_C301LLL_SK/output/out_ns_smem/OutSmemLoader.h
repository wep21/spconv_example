#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Ampere_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T1688_200_C301LLL_SK/output/out_ns_smem/tmap/Out5DLinear.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T1688_200_C301LLL_SK {
namespace output {
namespace out_ns_smem {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::conv::main::Ampere_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T1688_200_C301LLL_SK::output::out_ns_smem::tmap::Out5DLinear;
struct OutSmemLoader {
  tv::half_t * pointer_;
  __forceinline__ __device__  OutSmemLoader(tv::half_t * ptr, int thread_idx)   {
    auto thread_offset = ThreadMap::initial_offset(thread_idx);
    pointer_ = ptr + thread_offset[0] * 136 + thread_offset[1];
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<tv::half_t, 16, 0> & frag, int32_t pointer_offset)   {
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster) {
        TV_PRAGMA_UNROLL
        for (int group = 0; group < 1; ++group) {
            TV_PRAGMA_UNROLL
            for (int row = 0; row < 1; ++row) {
                const tv::half_t * cur_pointer =
                    pointer_ + row * 4 * 136 + group * 1 * 136 +
                    cluster * 1 * 136 + pointer_offset;
                int frag_row_idx =
                    (row + 1 * (group + 1 * cluster));
                tv::alignedarray<int4, 1, 16> *frag_ptr = reinterpret_cast<tv::alignedarray<int4, 1, 16> *>(&frag);
                tv::alignedarray<int4, 1, 16> const *memory_pointer =
                    reinterpret_cast<tv::alignedarray<int4, 1, 16> const *>(cur_pointer);
                TV_PRAGMA_UNROLL
                for (int column = 0; column < 2; ++column) {
                    int frag_idx = frag_row_idx * 2 + column;
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < 1; ++v) {
                        frag_ptr[frag_idx * 1 + v] =
                            memory_pointer[(column * 64 / 8) *
                                                1 +
                                            v];
                    }
                }
            }
        }
    }
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 16, 0> & frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    pointer_ += pointer_offset;
  }
};
} // namespace out_ns_smem
} // namespace output
} // namespace Ampere_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib