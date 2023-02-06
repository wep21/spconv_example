#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace gpAmpere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1 {
namespace iterb_p {
struct MaskTileIteratorParams {
  int32_t stride_;
  int64_t inc_strided_;
  int64_t inc_next_;
  __forceinline__ __host__ __device__  MaskTileIteratorParams()   {
    
  }
  __forceinline__ __host__ __device__  MaskTileIteratorParams(int stride) : stride_(stride)  {
    
    inc_strided_ = stride * 8 * sizeof(int8_t);
    inc_next_ = 64 - (1) *
                                    8 * stride *
                                    sizeof(int8_t);
  }
};
} // namespace iterb_p
} // namespace gpAmpere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib