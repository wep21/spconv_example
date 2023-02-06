#pragma once
#include <unordered_map>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu4d/spinds/ConvOutLocIter.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu4d/spinds/ConvProblem.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu4d/spinds64/ConvOutLocIter.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu4d {
using TensorView = spconvlib::cumm::common::TensorView;
using ConvLocIter = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::spinds::ConvOutLocIter;
using ConvProblem = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::spinds::ConvProblem;
using ConvLocIter64 = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::spinds64::ConvOutLocIter;
struct SparseConvIndicesCPU {
  /**
   * @param indices 
   * @param indice_pairs 
   * @param out_inds 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param input_dims 
   * @param ksize 
   * @param dilation 
   */
  static int generate_subm_conv_inds(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 4> input_dims, tv::array<int, 4> ksize, tv::array<int, 4> dilation);
  /**
   * @param indices 
   * @param indice_pairs 
   * @param out_inds 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   */
  static int generate_conv_inds(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 4> output_dims, tv::array<int, 4> input_dims, tv::array<int, 4> ksize, tv::array<int, 4> stride, tv::array<int, 4> padding, tv::array<int, 4> dilation, bool transposed = false);
};
} // namespace ops_cpu4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib