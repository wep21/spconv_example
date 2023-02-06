#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/gemm/layout/RowMajor.h>
#include <spconvlib/cumm/gemm/layout/ColumnMajor.h>
#include <spconvlib/cumm/gemm/main/SimtDP4A_s8s8s8s32s32tnt_m64n64k32m32n32k32A1_200_S1/mma/mma_ns_wmma/instmma/InstMma.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace SimtDP4A_s8s8s8s32s32tnt_m64n64k32m32n32k32A1_200_S1 {
namespace mma {
namespace mma_ns_wmma {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using RowMajor = spconvlib::cumm::gemm::layout::RowMajor;
using ColumnMajor = spconvlib::cumm::gemm::layout::ColumnMajor;
using InstMma = spconvlib::cumm::gemm::main::SimtDP4A_s8s8s8s32s32tnt_m64n64k32m32n32k32A1_200_S1::mma::mma_ns_wmma::instmma::InstMma;
struct WarpMmaSimt {
  __forceinline__ __device__ void operator()(tv::array<int32_t, 32, 0>& D, tv::array<int8_t, 32, 0> const & A, tv::array<int8_t, 16, 0> const & B, tv::array<int32_t, 32, 0> const & C)   {
    
    constexpr RowMajor layoutC = RowMajor::from_shape({8, 4});
    InstMma mma;
    D = C;
    tv::array<int8_t, 4, 0> const *ptr_A =
        reinterpret_cast<tv::array<int8_t, 4, 0> const *>(&A);
    tv::array<int8_t, 4, 0> const *ptr_B =
        reinterpret_cast<tv::array<int8_t, 4, 0> const *>(&B);
    TV_PRAGMA_UNROLL
    for (int k = 0; k < 1; ++k){
      TV_PRAGMA_UNROLL
      for (int n = 0; n < 4; ++n){
        TV_PRAGMA_UNROLL
        for (int m = 0; m < 8; ++m){
          tv::array<int32_t, 1, 0> tmp =
              reinterpret_cast<tv::array<int32_t, 1, 0> &>(D[layoutC(m, n)]);
          mma(tmp, ptr_A[m * 4 / 4 + k],
              ptr_B[n * 4 / 4 + k], tmp);
          D[layoutC(m, n)] = reinterpret_cast<int32_t &>(tmp);
        }
      }
    }
  }
};
} // namespace mma_ns_wmma
} // namespace mma
} // namespace SimtDP4A_s8s8s8s32s32tnt_m64n64k32m32n32k32A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib