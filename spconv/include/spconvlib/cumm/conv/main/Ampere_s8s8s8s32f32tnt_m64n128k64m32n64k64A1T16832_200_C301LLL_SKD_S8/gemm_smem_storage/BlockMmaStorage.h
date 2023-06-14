#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<int8_t, 8192, 16> smem_A;
  tv::alignedarray<int8_t, 16384, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Ampere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib