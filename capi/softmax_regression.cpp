#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/softmax_regression/softmax_regression_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetSoftmaxRegressionPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<SoftmaxRegression>(identifier,
      static_cast<SoftmaxRegression*>(value));
}

extern "C" void *mlpackGetSoftmaxRegressionPtr(const char* identifier)
{
  SoftmaxRegression *modelptr = GetParamPtr<SoftmaxRegression>(identifier);
  return modelptr;
}

static void SoftmaxRegressionMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackSoftmaxRegression()
{
  SoftmaxRegressionMlpackMain();
}

