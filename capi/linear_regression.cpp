#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/linear_regression/linear_regression_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetLinearRegressionPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<LinearRegression>(identifier,
      static_cast<LinearRegression*>(value));
}

extern "C" void *mlpackGetLinearRegressionPtr(const char* identifier)
{
  LinearRegression *modelptr = GetParamPtr<LinearRegression>(identifier);
  return modelptr;
}

static void LinearRegressionMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackLinearRegression()
{
  LinearRegressionMlpackMain();
}

