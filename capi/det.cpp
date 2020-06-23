#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/det/det_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetDTreePtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<DTree<>>(identifier,
      static_cast<DTree<>*>(value));
}

extern "C" void *mlpackGetDTreePtr(const char* identifier)
{
  DTree<> *modelptr = GetParamPtr<DTree<>>(identifier);
  return modelptr;
}

static void DetMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackDet()
{
  DetMlpackMain();
}

