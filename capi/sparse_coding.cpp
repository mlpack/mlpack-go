#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/sparse_coding/sparse_coding_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetSparseCodingPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<SparseCoding>(identifier,
      static_cast<SparseCoding*>(value));
}

extern "C" void *mlpackGetSparseCodingPtr(const char* identifier)
{
  SparseCoding *modelptr = GetParamPtr<SparseCoding>(identifier);
  return modelptr;
}

static void SparseCodingMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackSparseCoding()
{
  SparseCodingMlpackMain();
}

