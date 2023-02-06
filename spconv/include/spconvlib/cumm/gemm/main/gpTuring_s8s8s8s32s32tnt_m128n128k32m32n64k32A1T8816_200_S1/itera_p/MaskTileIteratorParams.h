#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace gpTuring_s8s8s8s32s32tnt_m128n128k32m32n64k32A1T8816_200_S1 {
namespace itera_p {
struct MaskTileIteratorParams {
  int32_t stride_;
  int64_t inc_strided_;
  int64_t inc_next_;
  int32_t const * indice_ptr_;
  __forceinline__ __host__ __device__  MaskTileIteratorParams()   {
    
  }
  __forceinline__ __host__ __device__  MaskTileIteratorParams(int stride, int32_t const * indice_ptr = nullptr) : stride_(stride), indice_ptr_(indice_ptr)  {
    
    inc_strided_ = stride * 16 * sizeof(int8_t);
    inc_next_ = 32 - (0) *
                                    16 * stride *
                                    sizeof(int8_t);
  }
};
} // namespace itera_p
} // namespace gpTuring_s8s8s8s32s32tnt_m128n128k32m32n64k32A1T8816_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib