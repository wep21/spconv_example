#pragma once
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu2d/spinds/ConvProblem.h>
#include <spconvlib/spconv/csrc/sparse/all/ops2d/spinds64/lociter/TensorGeneric.h>
#include <spconvlib/spconv/csrc/sparse/all/ops2d/spinds64/lociter_rs/TensorGeneric.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops2d {
namespace spinds64 {
using TensorView = spconvlib::cumm::common::TensorView;
using ConvProblem = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::spinds::ConvProblem;
using LayoutNPQ = spconvlib::spconv::csrc::sparse::all::ops2d::spinds64::lociter::TensorGeneric;
using LayoutRS = spconvlib::spconv::csrc::sparse::all::ops2d::spinds64::lociter_rs::TensorGeneric;
struct ConvOutLocIter {
  ConvProblem problem_;
  tv::array<int, 2> count_;
  LayoutNPQ layout_npq;
  LayoutRS layout_rs;
  TV_HOST_DEVICE_INLINE  ConvOutLocIter(ConvProblem const& problem) : problem_(problem), count_({0, 0}), layout_npq(LayoutNPQ::from_shape({problem.N, problem.output_dims[0], problem.output_dims[1]})), layout_rs(LayoutRS::from_shape({problem.ksize[0], problem.ksize[1]}))  {
    
  }
  TV_HOST_DEVICE_INLINE ConvOutLocIter& operator++()   {
    
    if (++count_[1] < problem_.ksize[1]){
        return *this;
    }
    count_[1] = 0;
    if (++count_[0] < problem_.ksize[0]){
        return *this;
    }
    count_[0] = 0;
    return *this;
  }
  TV_HOST_DEVICE_INLINE void set_filter_offset(int filter_offset)   {
    
    layout_rs.inverse(filter_offset, count_);
  }
  template <bool NoStride>
  TV_HOST_DEVICE_INLINE tv::array<int, 3> nhw_to_npq(const int* nhw_offset)  const {
    
    int r_0 = count_[0];
    int h_0 = (nhw_offset[1] + problem_.padding[0] - 
        r_0 * problem_.dilation[0]) / (NoStride ? 1 : problem_.stride[0]);
    int r_1 = count_[1];
    int h_1 = (nhw_offset[2] + problem_.padding[1] - 
        r_1 * problem_.dilation[1]) / (NoStride ? 1 : problem_.stride[1]);
    return {nhw_offset[0], h_0, h_1};
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 3> npq_to_nhw(const int* npq_offset)  const {
    
    int r_0 = count_[0];
    int h_0 = npq_offset[1] * problem_.stride[0] - problem_.padding[0] + r_0 * problem_.dilation[0];
    int r_1 = count_[1];
    int h_1 = npq_offset[2] * problem_.stride[1] - problem_.padding[1] + r_1 * problem_.dilation[1];
    return {npq_offset[0], h_0, h_1};
  }
  TV_HOST_DEVICE_INLINE bool query_npq(const int* nhw_offset, tv::array<int, 3>& npq_offset)  const {
    
    auto npq_no_stride = nhw_to_npq<true>(nhw_offset);
    npq_offset[0] = npq_no_stride[0];
    npq_offset[1] = npq_no_stride[1] / problem_.stride[0];
    npq_offset[2] = npq_no_stride[2] / problem_.stride[1];
    return npq_no_stride[0] < problem_.N && 
        npq_offset[1] >= 0 && npq_offset[1] < problem_.output_dims[0] && npq_offset[2] >= 0 && npq_offset[2] < problem_.output_dims[1] &&
        !(npq_no_stride[1] % problem_.stride[0]) && !(npq_no_stride[2] % problem_.stride[1]);
  }
  TV_HOST_DEVICE_INLINE bool query_npq_no_stride(const int* nhw_offset, tv::array<int, 3>& npq_offset)  const {
    
    npq_offset = nhw_to_npq<true>(nhw_offset);
    return npq_offset[0] < problem_.N && 
        npq_offset[1] >= 0 && npq_offset[1] < problem_.output_dims[0] && npq_offset[2] >= 0 && npq_offset[2] < problem_.output_dims[1];
  }
  TV_HOST_DEVICE_INLINE bool query_nhw(const int* npq_offset, tv::array<int, 3>& nhw_offset)  const {
    
    nhw_offset = npq_to_nhw(npq_offset);
    return nhw_offset[0] < problem_.N && 
        nhw_offset[1] >= 0 && nhw_offset[1] < problem_.input_dims[0] && nhw_offset[2] >= 0 && nhw_offset[2] < problem_.input_dims[1];
  }
  TV_HOST_DEVICE_INLINE bool query_nhw_out(const int* npq_offset, tv::array<int, 3>& nhw_offset)  const {
    
    nhw_offset = npq_to_nhw(npq_offset);
    return nhw_offset[0] < problem_.N && 
        nhw_offset[1] >= 0 && nhw_offset[1] < problem_.output_dims[0] && nhw_offset[2] >= 0 && nhw_offset[2] < problem_.output_dims[1];
  }
};
} // namespace spinds64
} // namespace ops2d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib