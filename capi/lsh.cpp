#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/lsh/lsh_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetLSHSearchPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<LSHSearch<>>(identifier,
      static_cast<LSHSearch<>*>(value));
}

extern "C" void *mlpackGetLSHSearchPtr(const char* identifier)
{
  LSHSearch<> *modelptr = GetParamPtr<LSHSearch<>>(identifier);
  return modelptr;
}

static void LshMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackLsh()
{
  LshMlpackMain();
}

