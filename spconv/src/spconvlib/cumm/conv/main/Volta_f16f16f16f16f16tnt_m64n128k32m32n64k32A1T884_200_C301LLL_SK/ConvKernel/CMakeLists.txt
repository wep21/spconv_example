
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_cumm_conv_main_Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_C301LLL_SK_ConvKernel OBJECT ConvKernel_conv_kernel.cu)
# set_target_properties(spconvlib_cumm_conv_main_Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_C301LLL_SK_ConvKernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_cumm_conv_main_Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_C301LLL_SK_ConvKernel PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_cumm_conv_main_Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_C301LLL_SK_ConvKernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_cumm_conv_main_Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_C301LLL_SK_ConvKernel PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_cumm_conv_main_Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_C301LLL_SK_ConvKernel PRIVATE
    /usr/local/cuda/include
    /home/daisuke/.local/lib/python3.10/site-packages/cumm/include)
target_compile_options(spconvlib_cumm_conv_main_Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_C301LLL_SK_ConvKernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -DTV_CUDA
    -gencode=arch=compute_86,code=[sm_86,compute_86]
    -Xcudafe "--diag_suppress=implicit_return_from_non_void_function"
    --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_cumm_conv_main_Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_C301LLL_SK_ConvKernel PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -O3
    -fPIC>)
                