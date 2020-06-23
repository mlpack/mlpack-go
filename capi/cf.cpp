#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/cf/cf_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetCFModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<CFModel>(identifier,
      static_cast<CFModel*>(value));
}

extern "C" void *mlpackGetCFModelPtr(const char* identifier)
{
  CFModel *modelptr = GetParamPtr<CFModel>(identifier);
  return modelptr;
}

static void CfMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackCf()
{
  CfMlpackMain();
}

