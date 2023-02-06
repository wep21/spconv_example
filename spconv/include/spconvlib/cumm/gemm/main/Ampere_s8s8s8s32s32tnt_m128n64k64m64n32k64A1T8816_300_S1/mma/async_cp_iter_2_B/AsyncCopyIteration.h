#pragma once
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_2_B/cp_async_copy/CpAsyncCopy.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1 {
namespace mma {
namespace async_cp_iter_2_B {
using CpAsyncCp = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_2_B::cp_async_copy::CpAsyncCopy;
struct AsyncCopyIteration {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {
    
    ///// nothing to do here /////
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {
    
    ///// nothing to do here /////
  }
};
} // namespace async_cp_iter_2_B
} // namespace mma
} // namespace Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib