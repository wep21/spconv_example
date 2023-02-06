#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/ops1d/SparseConvIndicesKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops2d/SparseConvIndicesKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops3d/SparseConvIndicesKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops4d/SparseConvIndicesKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
using ThrustCustomAllocatorV2 = spconvlib::spconv::csrc::sparse::all::ThrustCustomAllocatorV2;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using ThrustAllocator = spconvlib::spconv::csrc::sparse::alloc::ThrustAllocator;
using Point2Voxel1DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::Point2VoxelCPU;
using SpconvIndicesCPU1D = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::SparseConvIndicesCPU;
using Point2Voxel1D = spconvlib::spconv::csrc::sparse::all::ops1d::Point2Voxel;
using Point2Voxel2DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::Point2VoxelCPU;
using SpconvIndicesCPU2D = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::SparseConvIndicesCPU;
using Point2Voxel2D = spconvlib::spconv::csrc::sparse::all::ops2d::Point2Voxel;
using Point2Voxel3DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::Point2VoxelCPU;
using SpconvIndicesCPU3D = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::SparseConvIndicesCPU;
using Point2Voxel3D = spconvlib::spconv::csrc::sparse::all::ops3d::Point2Voxel;
using Point2Voxel4DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::Point2VoxelCPU;
using SpconvIndicesCPU4D = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::SparseConvIndicesCPU;
using Point2Voxel4D = spconvlib::spconv::csrc::sparse::all::ops4d::Point2Voxel;
using SpconvIndices1D = spconvlib::spconv::csrc::sparse::all::ops1d::SparseConvIndicesKernel;
using SpconvIndices2D = spconvlib::spconv::csrc::sparse::all::ops2d::SparseConvIndicesKernel;
using SpconvIndices3D = spconvlib::spconv::csrc::sparse::all::ops3d::SparseConvIndicesKernel;
using SpconvIndices4D = spconvlib::spconv::csrc::sparse::all::ops4d::SparseConvIndicesKernel;
int SpconvOps::generate_conv_inds_mask_stage2(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs_fwd, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_pairs_uniq_before_sort, tv::Tensor out_inds, tv::Tensor mask_fwd, tv::Tensor mask_bwd, int num_out_act, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed, std::uintptr_t stream_int)   {
  
  int ndim = indices.dim(1) - 1;
  TV_ASSERT_RT_ERR(output_dims.size() == ndim && input_dims.size() == ndim &&
      ksize.size() == ndim && stride.size() == ndim && dilation.size() == ndim &&
      padding.size() == ndim, "your params size not equal to ndim", ndim);
  if (ndim == 1){
      tv::array<int, 1> output_dims_, input_dims_;
      tv::array<int, 1> ksize_, stride_, padding_, dilation_;
      for (int i = 0; i < 1; ++i){
          output_dims_[i] = output_dims[i];
          input_dims_[i] = input_dims[i];
          ksize_[i] = ksize[i];
          stride_[i] = stride[i];
          padding_[i] = padding[i];
          dilation_[i] = dilation[i];
      }
      return SpconvIndices1D::generate_conv_inds_stage2_mask(
          indices, hashdata_k, hashdata_v,
          indice_pairs_fwd, indice_pairs_bwd, 
          indice_pairs_uniq, indice_pairs_uniq_before_sort,
          out_inds, mask_fwd, mask_bwd,
          num_out_act, batch_size, output_dims_, input_dims_, 
          ksize_, stride_, padding_, dilation_, transposed, stream_int);
  }
  if (ndim == 2){
      tv::array<int, 2> output_dims_, input_dims_;
      tv::array<int, 2> ksize_, stride_, padding_, dilation_;
      for (int i = 0; i < 2; ++i){
          output_dims_[i] = output_dims[i];
          input_dims_[i] = input_dims[i];
          ksize_[i] = ksize[i];
          stride_[i] = stride[i];
          padding_[i] = padding[i];
          dilation_[i] = dilation[i];
      }
      return SpconvIndices2D::generate_conv_inds_stage2_mask(
          indices, hashdata_k, hashdata_v,
          indice_pairs_fwd, indice_pairs_bwd, 
          indice_pairs_uniq, indice_pairs_uniq_before_sort,
          out_inds, mask_fwd, mask_bwd,
          num_out_act, batch_size, output_dims_, input_dims_, 
          ksize_, stride_, padding_, dilation_, transposed, stream_int);
  }
  if (ndim == 3){
      tv::array<int, 3> output_dims_, input_dims_;
      tv::array<int, 3> ksize_, stride_, padding_, dilation_;
      for (int i = 0; i < 3; ++i){
          output_dims_[i] = output_dims[i];
          input_dims_[i] = input_dims[i];
          ksize_[i] = ksize[i];
          stride_[i] = stride[i];
          padding_[i] = padding[i];
          dilation_[i] = dilation[i];
      }
      return SpconvIndices3D::generate_conv_inds_stage2_mask(
          indices, hashdata_k, hashdata_v,
          indice_pairs_fwd, indice_pairs_bwd, 
          indice_pairs_uniq, indice_pairs_uniq_before_sort,
          out_inds, mask_fwd, mask_bwd,
          num_out_act, batch_size, output_dims_, input_dims_, 
          ksize_, stride_, padding_, dilation_, transposed, stream_int);
  }
  if (ndim == 4){
      tv::array<int, 4> output_dims_, input_dims_;
      tv::array<int, 4> ksize_, stride_, padding_, dilation_;
      for (int i = 0; i < 4; ++i){
          output_dims_[i] = output_dims[i];
          input_dims_[i] = input_dims[i];
          ksize_[i] = ksize[i];
          stride_[i] = stride[i];
          padding_[i] = padding[i];
          dilation_[i] = dilation[i];
      }
      return SpconvIndices4D::generate_conv_inds_stage2_mask(
          indices, hashdata_k, hashdata_v,
          indice_pairs_fwd, indice_pairs_bwd, 
          indice_pairs_uniq, indice_pairs_uniq_before_sort,
          out_inds, mask_fwd, mask_bwd,
          num_out_act, batch_size, output_dims_, input_dims_, 
          ksize_, stride_, padding_, dilation_, transposed, stream_int);
  }
  TV_THROW_RT_ERR("unknown ndim", ndim);
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib