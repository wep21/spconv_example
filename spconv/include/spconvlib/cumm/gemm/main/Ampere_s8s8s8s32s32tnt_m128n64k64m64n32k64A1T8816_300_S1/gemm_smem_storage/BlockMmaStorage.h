#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<int8_t, 24576, 16> smem_A;
  tv::alignedarray<int8_t, 12288, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib