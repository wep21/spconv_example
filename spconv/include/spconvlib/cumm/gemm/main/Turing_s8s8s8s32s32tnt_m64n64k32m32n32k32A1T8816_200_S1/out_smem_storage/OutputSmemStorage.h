#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_s8s8s8s32s32tnt_m64n64k32m32n32k32A1T8816_200_S1 {
namespace out_smem_storage {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<int32_t, 1152, 16> smem;
};
} // namespace out_smem_storage
} // namespace Turing_s8s8s8s32s32tnt_m64n64k32m32n32k32A1T8816_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib