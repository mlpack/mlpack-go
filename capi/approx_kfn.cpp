#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/approx_kfn/approx_kfn_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetApproxKFNModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<ApproxKFNModel>(identifier,
      static_cast<ApproxKFNModel*>(value));
}

extern "C" void *mlpackGetApproxKFNModelPtr(const char* identifier)
{
  ApproxKFNModel *modelptr = GetParamPtr<ApproxKFNModel>(identifier);
  return modelptr;
}

static void ApproxKfnMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackApproxKfn()
{
  ApproxKfnMlpackMain();
}

