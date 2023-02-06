#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16ttt_m64n64k32m32n32k32A1T884_200_S1 {
namespace out_smem_storage {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<tv::half_t, 2176, 16> smem;
};
} // namespace out_smem_storage
} // namespace Volta_f16f16f16f16f16ttt_m64n64k32m32n32k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib