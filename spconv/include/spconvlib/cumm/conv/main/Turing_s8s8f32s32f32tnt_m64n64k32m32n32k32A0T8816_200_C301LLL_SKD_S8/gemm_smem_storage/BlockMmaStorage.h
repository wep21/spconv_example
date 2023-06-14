#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<int8_t, 4096, 16> smem_A;
  tv::alignedarray<int8_t, 4096, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib