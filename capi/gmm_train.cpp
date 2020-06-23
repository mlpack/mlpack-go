#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/gmm/gmm_train_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetGMMPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<GMM>(identifier,
      static_cast<GMM*>(value));
}

extern "C" void *mlpackGetGMMPtr(const char* identifier)
{
  GMM *modelptr = GetParamPtr<GMM>(identifier);
  return modelptr;
}

static void GmmTrainMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackGmmTrain()
{
  GmmTrainMlpackMain();
}

