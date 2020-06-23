#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/logistic_regression/logistic_regression_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetLogisticRegressionPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<LogisticRegression<>>(identifier,
      static_cast<LogisticRegression<>*>(value));
}

extern "C" void *mlpackGetLogisticRegressionPtr(const char* identifier)
{
  LogisticRegression<> *modelptr = GetParamPtr<LogisticRegression<>>(identifier);
  return modelptr;
}

static void LogisticRegressionMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackLogisticRegression()
{
  LogisticRegressionMlpackMain();
}

