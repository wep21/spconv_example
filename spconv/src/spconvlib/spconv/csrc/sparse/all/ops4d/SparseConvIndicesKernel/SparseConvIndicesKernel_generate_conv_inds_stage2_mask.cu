#include <spconvlib/spconv/csrc/sparse/all/ops4d/SparseConvIndicesKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops4d {
using TensorView = spconvlib::cumm::common::TensorView;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
using ConvLocIter = spconvlib::spconv::csrc::sparse::all::ops4d::spinds::ConvOutLocIter;
using ConvProblem = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::spinds::ConvProblem;
using ConvLocIter64 = spconvlib::spconv::csrc::sparse::all::ops4d::spinds64::ConvOutLocIter;
int SparseConvIndicesKernel::generate_conv_inds_stage2_mask(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs_fwd, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_pairs_uniq_before_sort, tv::Tensor out_inds, tv::Tensor mask_fwd, tv::Tensor mask_bwd, int num_out_act, int batch_size, tv::array<int, 4> output_dims, tv::array<int, 4> input_dims, tv::array<int, 4> ksize, tv::array<int, 4> stride, tv::array<int, 4> padding, tv::array<int, 4> dilation, bool transposed, std::uintptr_t stream_int)   {
  
  auto custream = reinterpret_cast<cudaStream_t>(stream_int);
  // TODO stream
  // TODO handle num input == 0
  int kv = ksize.op<tv::arrayops::prod>();
  // indice_pairs_bwd: [kv, num_act_in]  or empty
  // indice_pairs_fwd: [kv, num_act_out]
  auto ctx = tv::Context();
  ctx.set_cuda_stream(custream);
  int num_act_in = indices.dim(0);
  int num_act_out = num_out_act;
  TV_ASSERT_RT_ERR(hashdata_k.dtype() == indice_pairs_uniq.dtype(), "error");
  TV_ASSERT_RT_ERR(hashdata_v.dtype() == tv::int32, "error");
  // out_inds: [num_out_act, 5]
  // auto timer = tv::CudaContextTimer<>();
  if (!indice_pairs_bwd.empty()){
      tv::check_shape(indice_pairs_bwd, {kv, num_act_in});
  }
  tv::check_shape(indice_pairs_fwd, {kv, num_act_out});
  tv::check_shape(out_inds, {num_out_act, 5});
  tv::cuda::Launch launcher_num_act_in(num_act_in, custream);
  launcher_num_act_in.blocks.y = kv;
  tv::cuda::Launch launcher_num_act_in_no_y(num_act_in, custream);
  ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
  tv::cuda::Launch lanucher_build_hash(num_out_act, custream);
  bool use_int32 = problem.check_npq_not_overflow();
  // TODO handle invalid num_out_act
  indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_out_act);
  tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){
    using V = int32_t;
    using K = TV_DECLTYPE(I);
    using table_t =
        tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                    tv::hash::default_empty_key_v<K>, false>;
    TV_ASSERT_RT_ERR(hashdata_k.dim(0) >= num_out_act, "hash size not enough");
    table_t hash = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
    tv::hash::clear_map_split(hash, custream);
    if (int(use_int32) == 0){
      ConvLocIter64 loc_iter(problem);
      lanucher_build_hash(build_conv_hash_table<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, hash, 
          out_inds.data_ptr<int>(), indice_pairs_uniq.data_ptr<const K>(), 
          loc_iter.layout_npq, num_out_act);
    }
    else if (int(use_int32) == 1){
      ConvLocIter loc_iter(problem);
      lanucher_build_hash(build_conv_hash_table<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, hash, 
          out_inds.data_ptr<int>(), indice_pairs_uniq.data_ptr<const K>(), 
          loc_iter.layout_npq, num_out_act);
    }
    else{
      TV_THROW_RT_ERR("unknown val int(use_int32), available: [0, 1]")
    }
    if (!mask_bwd.empty()){
        launcher_num_act_in(calc_conv_indices_stage2_mask<table_t, false>, hash, 
            indice_pairs_fwd.data_ptr<int>(), indice_pairs_bwd.data_ptr<int>(), 
            indice_pairs_uniq_before_sort.data_ptr<K>(),
            mask_fwd.data_ptr<uint32_t>(), mask_bwd.data_ptr<uint32_t>(),
            num_act_in, indice_pairs_fwd.dim(1));
        launcher_num_act_in_no_y(calc_conv_indices_stage2_mask_output, 
            indice_pairs_bwd.data_ptr<int>(), 
            mask_bwd.data_ptr<uint32_t>(),
            num_act_in, kv);
        if (mask_fwd.dim(0) == 2){
            mask_fwd[1].copy_(mask_fwd[0], ctx);
        }
        if (mask_bwd.dim(0) == 2){
            mask_bwd[1].copy_(mask_bwd[0], ctx);
        }
    }else{
        launcher_num_act_in(calc_conv_indices_stage2_inference_mask<table_t, false>, hash, 
            indice_pairs_fwd.data_ptr<int>(), indice_pairs_bwd.data_ptr<int>(), 
            indice_pairs_uniq_before_sort.data_ptr<K>(),
            mask_fwd.data_ptr<uint32_t>(),
            num_act_in, indice_pairs_fwd.dim(1));
        if (mask_fwd.dim(0) == 2){
            mask_fwd[1].copy_(mask_fwd[0], ctx);
        }
    }
  });
  return num_out_act;
}
} // namespace ops4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib