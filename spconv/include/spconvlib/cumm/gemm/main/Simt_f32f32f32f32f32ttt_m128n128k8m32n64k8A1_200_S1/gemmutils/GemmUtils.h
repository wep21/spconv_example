#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1 {
namespace gemmutils {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct GemmUtils {
  __forceinline__ __host__ __device__ static int get_gemm_k_size_per_split(int k, int split_k)   {
    
    int total_gemm_k_iterations = tv::div_up(k, 8);
    int gemm_k_iterations_per_split =
        tv::div_up(total_gemm_k_iterations, split_k);
    auto gemm_k_size_per_split = gemm_k_iterations_per_split * 8; 
    return gemm_k_size_per_split;
  }
  __forceinline__ __host__ __device__ static int get_gemm_k_bound(int k, int gemm_k_size_per_split, int tile_offset_k)   {
    
    int k_bound = min(k, (tile_offset_k + 1) * gemm_k_size_per_split);
    return k_bound;
  }
  __forceinline__ __host__ __device__ static int get_gemm_iterations(int k_bound, int gemm_k_size_per_split, int tile_offset_k)   {
    
    int gemm_k_iterations =
        tv::div_up(k_bound - tile_offset_k * gemm_k_size_per_split, 8);
    return gemm_k_iterations;
  }
};
} // namespace gemmutils
} // namespace Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib