cmake_minimum_required(VERSION 3.20)
project(spconv_tensorrt_plugins)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(WARNING "cuda not found")
  return()
endif()

enable_language(CUDA)

find_package(CUDAToolkit)

set(CUMM_DISABLE_CMAKE_INSTALL ON CACHE BOOL "enable X functionality" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 75)

include(FetchContent)
FetchContent_Declare(cumm
  GIT_REPOSITORY https://github.com/FindDefinition/cumm
  GIT_TAG 92a3dfa7e3ad4f0d185fbe14a38c79adfb63d313 # v0.4.11
)

FetchContent_MakeAvailable(cumm)

add_subdirectory(spconv)

add_executable(main main.cu)
# SPCONV2_INCLUDE_PATH come from spconv/CMakeLists.txt
target_include_directories(main PRIVATE ${SPCONV2_INCLUDE_PATH})
target_link_libraries(main spconv cumm::cumm)


