#include </home/ryan/src/mlpack-3.3.2/src/mlpack/bindings/go/tests/test_go_binding_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetGaussianKernelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<GaussianKernel>(identifier,
      static_cast<GaussianKernel*>(value));
}

extern "C" void *mlpackGetGaussianKernelPtr(const char* identifier)
{
  GaussianKernel *modelptr = GetParamPtr<GaussianKernel>(identifier);
  return modelptr;
}

static void TestGoBindingMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackTestGoBinding()
{
  TestGoBindingMlpackMain();
}

