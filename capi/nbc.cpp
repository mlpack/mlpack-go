#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/naive_bayes/nbc_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetNBCModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<NBCModel>(identifier,
      static_cast<NBCModel*>(value));
}

extern "C" void *mlpackGetNBCModelPtr(const char* identifier)
{
  NBCModel *modelptr = GetParamPtr<NBCModel>(identifier);
  return modelptr;
}

static void NbcMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackNbc()
{
  NbcMlpackMain();
}

