#pragma once
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1/output/out_ns_frag/OutFragIter.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1/output/out_ns_warp/OutWarpTileIterator.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1/output/out_ns_smem/OutSmemLoader.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1/out_iter/OutIterator.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1/out_iter_const/OutIterator.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1/out_smem_storage/OutputSmemStorage.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1/out_op/LinearCombination.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1/output/out_ns_apply/ApplyOutputOp.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1 {
namespace output {
using FragIter = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1::output::out_ns_frag::OutFragIter;
using OutWarpIter = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1::output::out_ns_warp::OutWarpTileIterator;
using SmemLoader = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1::output::out_ns_smem::OutSmemLoader;
using OutIter = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1::out_iter::OutIterator;
using ConstOutIter = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1::out_iter_const::OutIterator;
using OutputStorage = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1::out_smem_storage::OutputSmemStorage;
using OutputOp = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1::out_op::LinearCombination;
using ApplyOp = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1::output::out_ns_apply::ApplyOutputOp;
struct Output {
  OutWarpIter warp_iter;
  SmemLoader smem_loader;
  __forceinline__ __device__  Output(OutputStorage* smem_storage, int thread_idx, int warp_idx_k, int warp_m, int warp_n, int lane_idx) : warp_iter(smem_storage->smem.data(), warp_idx_k * 4 + warp_m, warp_n, lane_idx), smem_loader(smem_storage->smem.data(), thread_idx)  {
    
  }
  __forceinline__ __device__ void run(OutputOp const& output_op, tv::array<float, 64, 0> const& accumulators, OutIter& out_iter, ConstOutIter& source_iter)   {
    
    if (!output_op.is_source_needed()){
      return run_no_source(output_op, accumulators, out_iter);
    }
    tv::array<float, 8, 0> source_frag;
    source_frag.clear();
    FragIter out_acc_iter(accumulators.data());
    TV_PRAGMA_UNROLL
    for (int iter = 0; iter < 8; iter += 1){
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        tv::array<float, 8, 0> acc_frag;
        out_acc_iter.load(acc_frag);
        ++out_acc_iter;
        warp_iter.store(acc_frag);
        if (p < 1 - 1){
            warp_iter.add_pointer_offset(2320);
        }
      }
      if (1 > 1){
          warp_iter.add_pointer_offset(0);
      }
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        source_iter.load(source_frag);
        ++source_iter;
        tv::array<float, 8, 0> smem_frags[1];
        smem_loader.load(smem_frags[0]);
        if (p < 1 - 1){
            smem_loader.add_pointer_offset(2320);
        }
        else if (1 > 1){
            TV_PRAGMA_UNROLL
            for (int partk_idx = 1; partk_idx < 1; ++partk_idx){
                smem_loader.add_pointer_offset(2320);
                smem_loader.load(smem_frags[partk_idx]);
                tv::math::plus<tv::array<float, 8, 0>> accer;
                smem_frags[0] = accer(smem_frags[0], smem_frags[partk_idx]);
            }
            smem_loader.add_pointer_offset(0);
        }
        tv::array<float, 8, 0> out_frag;
        ApplyOp::apply_output_operator(out_frag, output_op, smem_frags[0], source_frag);
        out_iter.store(out_frag);
        ++out_iter;
      }
      if (1 > 1){
          smem_loader.add_pointer_offset(0);
      }
    }
  }
  __forceinline__ __device__ void run_no_source(OutputOp const& output_op, tv::array<float, 64, 0> const& accumulators, OutIter& out_iter)   {
    
    FragIter out_acc_iter(accumulators.data());
    TV_PRAGMA_UNROLL
    for (int iter = 0; iter < 8; iter += 1){
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        tv::array<float, 8, 0> acc_frag;
        out_acc_iter.load(acc_frag);
        ++out_acc_iter;
        warp_iter.store(acc_frag);
        if (p < 1 - 1){
            warp_iter.add_pointer_offset(2320);
        }
      }
      if (1 > 1){
          warp_iter.add_pointer_offset(0);
      }
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        tv::array<float, 8, 0> smem_frags[1];
        smem_loader.load(smem_frags[0]);
        if (p < 1 - 1){
            smem_loader.add_pointer_offset(2320);
        }
        else if (1 > 1){
            TV_PRAGMA_UNROLL
            for (int partk_idx = 1; partk_idx < 1; ++partk_idx){
                smem_loader.add_pointer_offset(2320);
                smem_loader.load(smem_frags[partk_idx]);
                tv::math::plus<tv::array<float, 8, 0>> accer;
                smem_frags[0] = accer(smem_frags[0], smem_frags[partk_idx]);
            }
            smem_loader.add_pointer_offset(0);
        }
        tv::array<float, 8, 0> out_frag;
        ApplyOp::apply_output_operator_no_source(out_frag, output_op, smem_frags[0]);
        out_iter.store(out_frag);
        ++out_iter;
      }
      if (1 > 1){
          smem_loader.add_pointer_offset(0);
      }
    }
  }
  __forceinline__ __device__ void run_self_reduce(OutputOp const& output_op, tv::array<float, 64, 0> const& accumulators, OutIter& out_iter)   {
    
    tv::array<float, 8, 0> source_frag;
    source_frag.clear();
    FragIter out_acc_iter(accumulators.data());
    TV_PRAGMA_UNROLL
    for (int iter = 0; iter < 8; iter += 1){
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        tv::array<float, 8, 0> acc_frag;
        out_acc_iter.load(acc_frag);
        ++out_acc_iter;
        warp_iter.store(acc_frag);
        if (p < 1 - 1){
            warp_iter.add_pointer_offset(2320);
        }
      }
      if (1 > 1){
          warp_iter.add_pointer_offset(0);
      }
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        out_iter.load(source_frag);
        tv::array<float, 8, 0> smem_frags[1];
        smem_loader.load(smem_frags[0]);
        if (p < 1 - 1){
            smem_loader.add_pointer_offset(2320);
        }
        else if (1 > 1){
            TV_PRAGMA_UNROLL
            for (int partk_idx = 1; partk_idx < 1; ++partk_idx){
                smem_loader.add_pointer_offset(2320);
                smem_loader.load(smem_frags[partk_idx]);
                tv::math::plus<tv::array<float, 8, 0>> accer;
                smem_frags[0] = accer(smem_frags[0], smem_frags[partk_idx]);
            }
            smem_loader.add_pointer_offset(0);
        }
        tv::array<float, 8, 0> out_frag;
        ApplyOp::apply_output_operator(out_frag, output_op, smem_frags[0], source_frag);
        out_iter.store(out_frag);
        ++out_iter;
      }
      if (1 > 1){
          smem_loader.add_pointer_offset(0);
      }
    }
  }
};
} // namespace output
} // namespace Simt_f32f32f32f32f32ttt_m128n128k8m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib