#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/preprocess/preprocess_binarize_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

static void PreprocessBinarizeMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackPreprocessBinarize()
{
  PreprocessBinarizeMlpackMain();
}

