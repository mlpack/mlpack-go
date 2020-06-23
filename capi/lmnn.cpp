#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/lmnn/lmnn_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void LmnnMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackLmnn()
{
  LmnnMlpackMain();
}

