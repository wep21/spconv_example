#include <spconvlib/cumm/common/CompileInfo.h>
namespace spconvlib {
namespace cumm {
namespace common {
bool CompileInfo::gemm_algo_can_use_ptx(std::tuple<int, int> min_arch, std::tuple<int, int> arch)   {
  
  auto ptx_arch = std::make_tuple(8, 6);
  return min_arch <= ptx_arch && arch >= ptx_arch;
  return false;
}
} // namespace common
} // namespace cumm
} // namespace spconvlib