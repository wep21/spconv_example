#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8f16s32f32tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SK_S8 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<int8_t, 6144, 16> smem_A;
  tv::alignedarray<int8_t, 6144, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Ampere_s8s8f16s32f32tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib