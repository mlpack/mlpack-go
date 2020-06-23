#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/lars/lars_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetLARSPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<LARS>(identifier,
      static_cast<LARS*>(value));
}

extern "C" void *mlpackGetLARSPtr(const char* identifier)
{
  LARS *modelptr = GetParamPtr<LARS>(identifier);
  return modelptr;
}

static void LarsMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackLars()
{
  LarsMlpackMain();
}

