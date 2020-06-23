#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/neighbor_search/kfn_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetKFNModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<KFNModel>(identifier,
      static_cast<KFNModel*>(value));
}

extern "C" void *mlpackGetKFNModelPtr(const char* identifier)
{
  KFNModel *modelptr = GetParamPtr<KFNModel>(identifier);
  return modelptr;
}

static void KfnMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackKfn()
{
  KfnMlpackMain();
}

