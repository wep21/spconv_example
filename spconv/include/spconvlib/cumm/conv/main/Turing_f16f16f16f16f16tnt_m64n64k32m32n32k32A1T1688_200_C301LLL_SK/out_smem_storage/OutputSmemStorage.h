#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_f16f16f16f16f16tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK {
namespace out_smem_storage {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<tv::half_t, 1152, 16> smem;
};
} // namespace out_smem_storage
} // namespace Turing_f16f16f16f16f16tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib