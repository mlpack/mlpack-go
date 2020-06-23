#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/random_forest/random_forest_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetRandomForestModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<RandomForestModel>(identifier,
      static_cast<RandomForestModel*>(value));
}

extern "C" void *mlpackGetRandomForestModelPtr(const char* identifier)
{
  RandomForestModel *modelptr = GetParamPtr<RandomForestModel>(identifier);
  return modelptr;
}

static void RandomForestMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackRandomForest()
{
  RandomForestMlpackMain();
}

