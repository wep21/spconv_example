#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace gpAmpere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1 {
namespace itera_p {
struct MaskTileIteratorParams {
  int32_t stride_;
  int64_t inc_strided_;
  int64_t inc_next_;
  int32_t const * indice_ptr_;
  __forceinline__ __host__ __device__  MaskTileIteratorParams()   {
    
  }
  __forceinline__ __host__ __device__  MaskTileIteratorParams(int stride, int32_t const * indice_ptr = nullptr) : stride_(stride), indice_ptr_(indice_ptr)  {
    
    inc_strided_ = stride * 8 * sizeof(int8_t);
    inc_next_ = 64 - (3) *
                                    8 * stride *
                                    sizeof(int8_t);
  }
};
} // namespace itera_p
} // namespace gpAmpere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib