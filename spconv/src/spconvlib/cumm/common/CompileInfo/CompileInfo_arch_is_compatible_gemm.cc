#include <spconvlib/cumm/common/CompileInfo.h>
namespace spconvlib {
namespace cumm {
namespace common {
bool CompileInfo::arch_is_compatible_gemm(std::tuple<int, int> arch)   {
  
  if (arch > std::make_tuple(8, 6)){
      return true;
  }
  return arch_is_compiled_gemm(arch);
}
} // namespace common
} // namespace cumm
} // namespace spconvlib