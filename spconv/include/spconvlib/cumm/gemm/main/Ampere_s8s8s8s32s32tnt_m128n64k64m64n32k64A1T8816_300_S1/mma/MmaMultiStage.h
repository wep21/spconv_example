#pragma once
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/mma_ns_miter/MaskIGemmIterator.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/mma_ns_miterD/MaskIGemmIteratorMaskLoaderDynamic.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/mma_ns_wa/WarpIteratorCrosswise.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/mma_ns_wb/WarpIteratorCrosswise.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/mma_ns_sa/SmemTileIterator.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/mma_ns_sb/SmemTileIterator.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/gemm_smem_storage/BlockMmaStorage.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/cpasync_group/CpAsyncGroup.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_global_A/AsyncCopyIteration.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_global_B/AsyncCopyIteration.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_0_A/AsyncCopyIteration.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_0_B/AsyncCopyIteration.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_1_A/AsyncCopyIteration.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_1_B/AsyncCopyIteration.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_2_A/AsyncCopyIteration.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_2_B/AsyncCopyIteration.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_3_A/AsyncCopyIteration.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/async_cp_iter_3_B/AsyncCopyIteration.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/inpitera/MaskTileIterator.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/inpiterb/MaskTileIterator.h>
#include <spconvlib/cumm/gemm/main/Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1/mma/mma_ns_wmma/WarpMmaTuring.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1 {
namespace mma {
using MaskIGemmIterator = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::mma_ns_miter::MaskIGemmIterator;
using MaskIGemmIteratorDynamic = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::mma_ns_miterD::MaskIGemmIteratorMaskLoaderDynamic;
using WarpIterA = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::mma_ns_wa::WarpIteratorCrosswise;
using WarpIterB = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::mma_ns_wb::WarpIteratorCrosswise;
using SmemIterA = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::mma_ns_sa::SmemTileIterator;
using SmemIterB = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::mma_ns_sb::SmemTileIterator;
using GemmStorage = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::gemm_smem_storage::BlockMmaStorage;
using CpAsyncGroup = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::cpasync_group::CpAsyncGroup;
using GlobalAsyncCopyIter_A = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_global_A::AsyncCopyIteration;
using GlobalAsyncCopyIter_B = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_global_B::AsyncCopyIteration;
using AsyncCopyIter_0_A = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_0_A::AsyncCopyIteration;
using AsyncCopyIter_0_B = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_0_B::AsyncCopyIteration;
using AsyncCopyIter_1_A = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_1_A::AsyncCopyIteration;
using AsyncCopyIter_1_B = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_1_B::AsyncCopyIteration;
using AsyncCopyIter_2_A = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_2_A::AsyncCopyIteration;
using AsyncCopyIter_2_B = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_2_B::AsyncCopyIteration;
using AsyncCopyIter_3_A = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_3_A::AsyncCopyIteration;
using AsyncCopyIter_3_B = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::async_cp_iter_3_B::AsyncCopyIteration;
using InputIteratorA = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::inpitera::MaskTileIterator;
using InputIteratorB = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::inpiterb::MaskTileIterator;
using WarpMma = spconvlib::cumm::gemm::main::Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1::mma::mma_ns_wmma::WarpMmaTuring;
struct MmaMultiStage {
  WarpIterA warp_iter_A;
  WarpIterB warp_iter_B;
  SmemIterA smem_iter_A;
  SmemIterB smem_iter_B;
  __forceinline__ __device__  MmaMultiStage(GemmStorage* smem_storage, int thread_idx, int warp_idx_k, int warp_m, int warp_n, int lane_idx) : warp_iter_A(smem_storage->smem_A.data(), warp_idx_k, warp_m, lane_idx), warp_iter_B(smem_storage->smem_B.data(), warp_idx_k, warp_n, lane_idx), smem_iter_A(128, smem_storage->smem_A.data(), thread_idx), smem_iter_B(64, smem_storage->smem_B.data(), thread_idx)  {
    
  }
  __forceinline__ __device__ void copy_tiles_and_advance(InputIteratorA & input_iter_A, InputIteratorB & input_iter_B, const int & group_idx)   {
    
    #if (defined(DEBUG_MMA_MS_DOWNFALL_A) || defined(DEBUG_MMA_MS_DOWNFALL_B))
                      if (group_idx == 3){
      #ifdef DEBUG_MMA_MS_DOWNFALL_A
                      tv::array<int8_t, 64, 0> input_frag_A;
      #ifndef DEBUG_MMA_MS_NOT_READ_INPUT_A
                          input_iter_A.load(input_frag_A);
      #endif
      #ifndef DEBUG_MMA_MS_NOT_WRITE_SMEM_A
                          smem_iter_A.store(input_frag_A);
      #endif
      #endif
      #ifdef DEBUG_MMA_MS_DOWNFALL_B
                      tv::array<int8_t, 32, 0> input_frag_B;
      #ifndef DEBUG_MMA_MS_NOT_READ_INPUT_B
                          input_iter_B.load(input_frag_B);
      #endif
      #ifndef DEBUG_MMA_MS_NOT_WRITE_SMEM_B
                          smem_iter_B.store(input_frag_B);
      #endif
      #endif
                      }
    #endif
                        if(group_idx == 0){
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_A) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_A))
                            AsyncCopyIter_0_A::do_copy(input_iter_A, smem_iter_A);
    #endif
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_B) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_B))
                            AsyncCopyIter_0_B::do_copy(input_iter_B, smem_iter_B);
    #endif
                            return;
                        }
                        if(group_idx == 1){
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_A) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_A))
                            AsyncCopyIter_1_A::do_copy(input_iter_A, smem_iter_A);
    #endif
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_B) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_B))
                            AsyncCopyIter_1_B::do_copy(input_iter_B, smem_iter_B);
    #endif
                            return;
                        }
                        if(group_idx == 2){
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_A) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_A))
                            AsyncCopyIter_2_A::do_copy(input_iter_A, smem_iter_A);
    #endif
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_B) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_B))
                            AsyncCopyIter_2_B::do_copy(input_iter_B, smem_iter_B);
    #endif
                            return;
                        }
                        if(group_idx == 3){
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_A) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_A))
                            AsyncCopyIter_3_A::do_copy(input_iter_A, smem_iter_A);
    #endif
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_B) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_B))
                            AsyncCopyIter_3_B::do_copy(input_iter_B, smem_iter_B);
    #endif
                            return;
                        }
  }
  __forceinline__ __device__ void operator()(int gemm_k_iterations, tv::array<int32_t, 64, 0>& accumulators, InputIteratorA & input_iter_A, InputIteratorB & input_iter_B, tv::array<int32_t, 64, 0> const& src_accumulators)   {
    
    accumulators = src_accumulators;
    TV_PRAGMA_UNROLL
    for(int stage = 0; stage< 2; ++stage, --gemm_k_iterations){
        if(gemm_k_iterations == 0){
            input_iter_A.clear_mask();
            input_iter_B.clear_mask();
        }
        GlobalAsyncCopyIter_A::do_copy_zfill(input_iter_A, smem_iter_A);
        GlobalAsyncCopyIter_B::do_copy_zfill(input_iter_B, smem_iter_B);
        ++input_iter_A;
        ++input_iter_B;
        ++smem_iter_A;
        ++smem_iter_B;
        CpAsyncGroup::make_fence();
    }
    CpAsyncGroup::wait_final_group();
    __syncthreads();
    tv::array<int8_t, 32, 0> warp_frag_A[2];
    tv::array<int8_t, 16, 0> warp_frag_B[2];
    warp_iter_A.set_kgroup_index(0);
    warp_iter_B.set_kgroup_index(0);
    warp_iter_A.load(warp_frag_A[0]);
    warp_iter_B.load(warp_frag_B[0]);
    ++warp_iter_A;
    ++warp_iter_B;
    WarpMma warp_mma;
    int smem_write_stage_idx = 2;
    int smem_read_stage_idx = 0;
    if (gemm_k_iterations == 0) {
        input_iter_A.clear_mask();
        input_iter_B.clear_mask();
    }
    for (; gemm_k_iterations > -2; ){
      TV_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < 4; ++warp_mma_k){
        warp_iter_A.set_kgroup_index((warp_mma_k + 1) % 4);
        warp_iter_B.set_kgroup_index((warp_mma_k + 1) % 4);
        // if warp_mma_k is last, smem load next, warp load next too.
        warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);
        ++warp_iter_A;
        ++warp_iter_B;
        // if (warp_mma_k > 0)
        //     warp_mma.transform(...)
        warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
            warp_frag_B[warp_mma_k % 2], accumulators);
        if (warp_mma_k < 3)
            copy_tiles_and_advance(input_iter_A, input_iter_B, warp_mma_k);
        if (warp_mma_k + 2 == 4) {
            copy_tiles_and_advance(input_iter_A, input_iter_B, 3);
            CpAsyncGroup::make_fence();
            CpAsyncGroup::wait_final_group();
            __syncthreads();
            ++smem_iter_A;
            ++smem_iter_B;
            ++input_iter_A;
            ++input_iter_B;
            if (smem_write_stage_idx == 2) {
                smem_iter_A.tile_increment(-3);
                smem_iter_B.tile_increment(-3);
                smem_write_stage_idx = 0;
            } else
                ++smem_write_stage_idx;
            if (smem_read_stage_idx == 2) {
                warp_iter_A.tile_increment(-3 *
                                        4);
                warp_iter_B.tile_increment(-3 *
                                        4);
                smem_read_stage_idx = 0;
            } else
                ++smem_read_stage_idx;
            --gemm_k_iterations;
            if (gemm_k_iterations == 0){
                input_iter_A.clear_mask();
                input_iter_B.clear_mask();
            }
        }
      }
    }
  }
};
} // namespace mma
} // namespace Ampere_s8s8s8s32s32tnt_m128n64k64m64n32k64A1T8816_300_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib