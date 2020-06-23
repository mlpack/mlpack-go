#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/emst/emst_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void EmstMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackEmst()
{
  EmstMlpackMain();
}

