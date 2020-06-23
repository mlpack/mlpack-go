#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/kernel_pca/kernel_pca_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void KernelPcaMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackKernelPca()
{
  KernelPcaMlpackMain();
}

