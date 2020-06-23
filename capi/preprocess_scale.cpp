#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/preprocess/preprocess_scale_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetScalingModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<ScalingModel>(identifier,
      static_cast<ScalingModel*>(value));
}

extern "C" void *mlpackGetScalingModelPtr(const char* identifier)
{
  ScalingModel *modelptr = GetParamPtr<ScalingModel>(identifier);
  return modelptr;
}

static void PreprocessScaleMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackPreprocessScale()
{
  PreprocessScaleMlpackMain();
}

