package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_preprocess_one_hot_encoding
#include <capi/preprocess_one_hot_encoding.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type PreprocessOneHotEncodingOptionalParam struct {
    Verbose bool
}

func PreprocessOneHotEncodingOptions() *PreprocessOneHotEncodingOptionalParam {
  return &PreprocessOneHotEncodingOptionalParam{
    Verbose: false,
  }
}

/*
  This utility takes a dataset and a vector of indices and does one-hot encoding
  of the respective features at those indices. Indices represent the IDs of the
  dimensions to be one-hot encoded.
  
  The output matrix with encoded features may be saved with the "Output"
  parameters.

  So, a simple example where we want to encode 1st and 3rd feature from dataset
  X into X_output would be
  
  // Initialize optional parameters for PreprocessOneHotEncoding().
  param := mlpack.PreprocessOneHotEncodingOptions()
  
  X_ouput := mlpack.PreprocessOneHotEncoding(X, 1, 3, param)

  Input parameters:

   - dimensions ([]int): Index of dimensions thatneed to be one-hot
        encoded.
   - input (mat.Dense): Matrix containing data.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix to save one-hot encoded features data to.

 */
func PreprocessOneHotEncoding(dimensions []int, input *mat.Dense, param *PreprocessOneHotEncodingOptionalParam) (*mat.Dense) {
  params := getParams("preprocess_one_hot_encoding")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  setParamVecInt(params, "dimensions", dimensions)
  setPassed(params, "dimensions")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat(params, "input", input, false)
  setPassed(params, "input")

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "output")

  // Call the mlpack program.
  C.mlpackPreprocessOneHotEncoding(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat(params, "output")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return output
}
