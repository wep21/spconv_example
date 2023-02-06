#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1 {
namespace iterb_p {
struct MaskTileIteratorParams {
  int32_t stride_;
  int64_t inc_strided_;
  int64_t inc_next_;
  __forceinline__ __host__ __device__  MaskTileIteratorParams()   {
    
  }
  __forceinline__ __host__ __device__  MaskTileIteratorParams(int stride) : stride_(stride)  {
    
    inc_strided_ = stride * 4 * sizeof(tv::half_t);
    inc_next_ = 128 - (1) *
                                    4 * stride *
                                    sizeof(tv::half_t);
  }
};
} // namespace iterb_p
} // namespace gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib