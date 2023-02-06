#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace gpSimtDP4A_s8s8s8s32s32tnt_m128n64k32m64n32k32A1_200_S1 {
namespace iterb_p {
struct MaskTileIteratorParams {
  int32_t stride_;
  int64_t inc_strided_;
  int64_t inc_next_;
  __forceinline__ __host__ __device__  MaskTileIteratorParams()   {
    
  }
  __forceinline__ __host__ __device__  MaskTileIteratorParams(int stride) : stride_(stride)  {
    
    inc_strided_ = stride * 64 * sizeof(int8_t);
    inc_next_ = 32 - (0) *
                                    64 * stride *
                                    sizeof(int8_t);
  }
};
} // namespace iterb_p
} // namespace gpSimtDP4A_s8s8s8s32s32tnt_m128n64k32m64n32k32A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib