#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace SimtDP4A_s8s8s8s32s32tnt_m128n64k32m64n32k32A1_200_S1 {
namespace out_smem_storage {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<int32_t, 1296, 16> smem;
};
} // namespace out_smem_storage
} // namespace SimtDP4A_s8s8s8s32s32tnt_m128n64k32m64n32k32A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib