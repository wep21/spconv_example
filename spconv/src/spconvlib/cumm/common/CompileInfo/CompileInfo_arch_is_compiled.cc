#include <spconvlib/cumm/common/CompileInfo.h>
namespace spconvlib {
namespace cumm {
namespace common {
bool CompileInfo::arch_is_compiled(std::tuple<int, int> arch)   {
  
  
  if (arch == std::make_tuple(8, 6)){
      return true;
  }
  return false;
}
} // namespace common
} // namespace cumm
} // namespace spconvlib