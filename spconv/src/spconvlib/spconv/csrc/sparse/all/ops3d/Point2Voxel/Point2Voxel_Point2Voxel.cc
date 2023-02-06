#include <spconvlib/spconv/csrc/sparse/all/ops3d/Point2Voxel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
using TensorView = spconvlib::cumm::common::TensorView;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops3d::p2v_c::Point2VoxelCommon;
using Layout = spconvlib::spconv::csrc::sparse::all::ops3d::layout_ns::TensorGeneric;
 Point2Voxel::Point2Voxel(std::array<float, 3> vsize_xyz, std::array<float, 6> coors_range_xyz, int num_point_features, int max_num_voxels, int max_num_points_per_voxel)   {
  
  for (int i = 0; i < 3; ++i){
      vsize[2 - i] = vsize_xyz[i];
      coors_range[2 - i] = coors_range_xyz[i];
      coors_range[5 - i] = coors_range_xyz[i + 3];
  }
  int64_t prod = 1;
  for (size_t i = 0; i < 3; ++i) {
      grid_size[i] =
          std::round((coors_range[3 + i] - coors_range[i]) / vsize[i]);
  }
  for (int i = 3 - 1; i >= 0; --i) {
      grid_stride[i] = prod;
      prod *= grid_size[i];
  }
  voxels = tv::zeros({max_num_voxels, max_num_points_per_voxel, num_point_features}, tv::type_v<float>, 0);
  indices = tv::zeros({max_num_voxels, 3}, tv::int32, 0);
  num_per_voxel = tv::zeros({max_num_voxels}, tv::int32, 0);
  hashdata = tv::zeros({1}, tv::custom128, 0);
  point_indice_data = tv::zeros({1}, tv::int64, 0);
}
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib