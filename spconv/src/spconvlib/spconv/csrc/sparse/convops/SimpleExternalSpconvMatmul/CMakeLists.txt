
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_spconv_csrc_sparse_convops_SimpleExternalSpconvMatmul OBJECT SimpleExternalSpconvMatmul_SimpleExternalSpconvMatmul.cc SimpleExternalSpconvMatmul_dtor_pccm_destructorSimpleExternalSpconvMatmul.cc SimpleExternalSpconvMatmul_check_cublas_status.cc SimpleExternalSpconvMatmul_tv_dtype_to_blaslt.cc SimpleExternalSpconvMatmul_matmul_colmajor.cc SimpleExternalSpconvMatmul_matmul.cc SimpleExternalSpconvMatmul_indice_conv_init_gemm.cc)
# set_target_properties(spconvlib_spconv_csrc_sparse_convops_SimpleExternalSpconvMatmul PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_spconv_csrc_sparse_convops_SimpleExternalSpconvMatmul PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_spconv_csrc_sparse_convops_SimpleExternalSpconvMatmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_spconv_csrc_sparse_convops_SimpleExternalSpconvMatmul PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_spconv_csrc_sparse_convops_SimpleExternalSpconvMatmul PRIVATE
    /usr/local/cuda/include
    /home/daisuke/.local/lib/python3.10/site-packages/cumm/include)
target_compile_options(spconvlib_spconv_csrc_sparse_convops_SimpleExternalSpconvMatmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -DTV_CUDA
    -gencode=arch=compute_86,code=[sm_86,compute_86]
    -Xcudafe "--diag_suppress=implicit_return_from_non_void_function"
    --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_spconv_csrc_sparse_convops_SimpleExternalSpconvMatmul PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -DTV_CUDA
    -O3
    -fPIC>)
                