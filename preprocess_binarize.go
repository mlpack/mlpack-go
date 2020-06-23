package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_preprocess_binarize
#include <capi/preprocess_binarize.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type PreprocessBinarizeOptionalParam struct {
    Dimension int
    Threshold float64
    Verbose bool
}

func PreprocessBinarizeOptions() *PreprocessBinarizeOptionalParam {
  return &PreprocessBinarizeOptionalParam{
    Dimension: 0,
    Threshold: 0,
    Verbose: false,
  }
}

/*
  This utility takes a dataset and binarizes the variables into either 0 or 1
  given threshold. User can apply binarization on a dimension or the whole
  dataset.  The dimension to apply binarization to can be specified using the
  "Dimension" parameter; if left unspecified, every dimension will be binarized.
   The threshold for binarization can also be specified with the "Threshold"
  parameter; the default threshold is 0.0.
  
  The binarized matrix may be saved with the "Output" output parameter.
  
  For example, if we want to set all variables greater than 5 in the dataset X
  to 1 and variables less than or equal to 5.0 to 0, and save the result to Y,
  we could run
  
  // Initialize optional parameters for PreprocessBinarize().
  param := mlpack.PreprocessBinarizeOptions()
  param.Threshold = 5
  
  Y := mlpack.PreprocessBinarize(X, param)
  
  But if we want to apply this to only the first (0th) dimension of X,  we could
  instead run
  
  // Initialize optional parameters for PreprocessBinarize().
  param := mlpack.PreprocessBinarizeOptions()
  param.Threshold = 5
  param.Dimension = 0
  
  Y := mlpack.PreprocessBinarize(X, param)


  Input parameters:

   - input (mat.Dense): Input data matrix.
   - Dimension (int): Dimension to apply the binarization. If not set, the
        program will binarize every dimension by default.  Default value 0.
   - Threshold (float64): Threshold to be applied for binarization. If not
        set, the threshold defaults to 0.0.  Default value 0.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix in which to save the output.

 */
func PreprocessBinarize(input *mat.Dense, param *PreprocessBinarizeOptionalParam) (*mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Binarize Data")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Dimension != 0 {
    setParamInt("dimension", param.Dimension)
    setPassed("dimension")
  }

  // Detect if the parameter was passed; set if so.
  if param.Threshold != 0 {
    setParamDouble("threshold", param.Threshold)
    setPassed("threshold")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("output")

  // Call the mlpack program.
  C.mlpackPreprocessBinarize()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat("output")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return output
}
