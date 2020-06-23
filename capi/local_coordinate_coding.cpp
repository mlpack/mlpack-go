#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/local_coordinate_coding/local_coordinate_coding_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetLocalCoordinateCodingPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<LocalCoordinateCoding>(identifier,
      static_cast<LocalCoordinateCoding*>(value));
}

extern "C" void *mlpackGetLocalCoordinateCodingPtr(const char* identifier)
{
  LocalCoordinateCoding *modelptr = GetParamPtr<LocalCoordinateCoding>(identifier);
  return modelptr;
}

static void LocalCoordinateCodingMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackLocalCoordinateCoding()
{
  LocalCoordinateCodingMlpackMain();
}

