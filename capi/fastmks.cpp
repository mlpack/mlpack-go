#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/fastmks/fastmks_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetFastMKSModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<FastMKSModel>(identifier,
      static_cast<FastMKSModel*>(value));
}

extern "C" void *mlpackGetFastMKSModelPtr(const char* identifier)
{
  FastMKSModel *modelptr = GetParamPtr<FastMKSModel>(identifier);
  return modelptr;
}

static void FastmksMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackFastmks()
{
  FastmksMlpackMain();
}

