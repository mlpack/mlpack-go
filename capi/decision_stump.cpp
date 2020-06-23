#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/decision_stump/decision_stump_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetDSModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<DSModel>(identifier,
      static_cast<DSModel*>(value));
}

extern "C" void *mlpackGetDSModelPtr(const char* identifier)
{
  DSModel *modelptr = GetParamPtr<DSModel>(identifier);
  return modelptr;
}

static void DecisionStumpMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackDecisionStump()
{
  DecisionStumpMlpackMain();
}

