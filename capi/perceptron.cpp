#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/perceptron/perceptron_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetPerceptronModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<PerceptronModel>(identifier,
      static_cast<PerceptronModel*>(value));
}

extern "C" void *mlpackGetPerceptronModelPtr(const char* identifier)
{
  PerceptronModel *modelptr = GetParamPtr<PerceptronModel>(identifier);
  return modelptr;
}

static void PerceptronMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackPerceptron()
{
  PerceptronMlpackMain();
}

