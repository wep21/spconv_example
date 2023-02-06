#pragma once
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n128k64m64n64k64A1T8816_300_S1/mma/async_cp_iter_3_A/cp_async_copy/CpAsyncCopy.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Ampere_s8s8s8s32s32tnt_m128n128k64m64n64k64A1T8816_300_S1 {
namespace mma {
namespace async_cp_iter_3_A {
using CpAsyncCp = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n128k64m64n64k64A1T8816_300_S1::mma::async_cp_iter_3_A::cp_async_copy::CpAsyncCopy;
struct AsyncCopyIteration {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {
    
    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(3, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(3, 0, valid);
    CpAsyncCp::copy(dest_ptr, src_ptr, valid);
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {
    
    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(3, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(3, 0, valid);
    CpAsyncCp::copy_zfill(dest_ptr, src_ptr, valid);
  }
};
} // namespace async_cp_iter_3_A
} // namespace mma
} // namespace Ampere_s8s8s8s32s32tnt_m128n128k64m64n64k64A1T8816_300_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib