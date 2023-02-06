#pragma once
#include <spconvlib/cumm/gemm/main/gpTuring_s8s8s8s32s32tnt_m64n64k32m32n32k32A1T8816_200_S1/out_params_ns/tmap/Out5DLinear.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace gpTuring_s8s8s8s32s32tnt_m64n64k32m32n32k32A1T8816_200_S1 {
namespace out_params_ns {
using ThreadMap = spconvlib::cumm::gemm::main::gpTuring_s8s8s8s32s32tnt_m64n64k32m32n32k32A1T8816_200_S1::out_params_ns::tmap::Out5DLinear;
struct OutIteratorParams {
  int64_t stride;
  int64_t increment_row;
  int64_t increment_group;
  int64_t increment_cluster;
  int64_t advance_row;
  int64_t advance_group;
  int64_t advance_cluster;
  int64_t advance_tile;
  const int* indice_ptr_;
  __forceinline__ __host__ __device__  OutIteratorParams()   {
    
  }
  __forceinline__ __host__ __device__  OutIteratorParams(int stride_, const int* indice_ptr = nullptr) : stride(stride_), indice_ptr_(indice_ptr)  {
    
    auto increment_params = ThreadMap::iteration_inc_params(stride);
    auto advance_params = ThreadMap::iteration_advance_params(stride);
    increment_cluster = increment_params[0];
    increment_group = increment_params[1];
    increment_row = increment_params[2];
    advance_tile = advance_params[0];
    advance_cluster = advance_params[1];
    advance_group = advance_params[2];
    advance_row = advance_params[3];
  }
};
} // namespace out_params_ns
} // namespace gpTuring_s8s8s8s32s32tnt_m64n64k32m32n32k32A1T8816_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib