#include <spconvlib/cumm/common/CompileInfo.h>
namespace spconvlib {
namespace cumm {
namespace common {
std::vector<std::tuple<int, int>> CompileInfo::get_compiled_gemm_cuda_arch()   {
  
  std::vector<std::tuple<int, int>> res;
  res.push_back(std::make_tuple(8, 6));
  return res;
}
} // namespace common
} // namespace cumm
} // namespace spconvlib