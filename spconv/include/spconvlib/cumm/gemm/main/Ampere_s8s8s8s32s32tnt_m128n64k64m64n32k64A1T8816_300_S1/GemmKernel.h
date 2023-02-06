#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTCKernel.h>
#include <spconvlib/cumm/gemm/layout/RowMajor.h>
#include <spconvlib/cumm/gemm/layout/ColumnMajor.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/utils/GemmUtilsCPU.h>
#include <spconvlib/cumm/common/GemmKernelFlags.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/gemmutils/GemmUtils.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/inpitera/MaskTileIterator.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/inpiterb/MaskTileIterator.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/gemm_smem_storage/BlockMmaStorage.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/out_smem_storage/OutputSmemStorage.h>
#include <spconvlib/cumm/gemm/main/gpAmpere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/out_iter/OutIterator.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/out_iter_const/OutIterator.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/out_op/LinearCombination.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/MmaMultiStage.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/output/Output.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1 {
using TensorViewNVRTCKernel = spconvlib::cumm::common::TensorViewNVRTCKernel;
using RowMajor = spconvlib::cumm::gemm::layout::RowMajor;
using ColumnMajor = spconvlib::cumm::gemm::layout::ColumnMajor;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using GemmUtilsCPU = spconvlib::cumm::gemm::utils::GemmUtilsCPU;
using GemmKernelFlags = spconvlib::cumm::common::GemmKernelFlags;
using GemmUtils = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::gemmutils::GemmUtils;
using InputIteratorA = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::inpitera::MaskTileIterator;
using InputIteratorB = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::inpiterb::MaskTileIterator;
using BlockMmaStorage = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::gemm_smem_storage::BlockMmaStorage;
using OutputSmemStorage = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::out_smem_storage::OutputSmemStorage;
using GemmParams = spconvlib::cumm::gemm::main::gpAmpere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::GemmParams;
using OutIter = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::out_iter::OutIterator;
using ConstOutIter = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::out_iter_const::OutIterator;
using OutputOp = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::out_op::LinearCombination;
using Mma = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::MmaMultiStage;
using Output = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::output::Output;
/**
 * @param params 
 */
__global__ void gemm_kernel(GemmParams params);
struct GemmKernel {
};
} // namespace Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib