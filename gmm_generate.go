package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_gmm_generate
#include <capi/gmm_generate.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type GmmGenerateOptionalParam struct {
    Seed int
    Verbose bool
}

func GmmGenerateOptions() *GmmGenerateOptionalParam {
  return &GmmGenerateOptionalParam{
    Seed: 0,
    Verbose: false,
  }
}

/*
  This program is able to generate samples from a pre-trained GMM (use gmm_train
  to train a GMM).  The pre-trained GMM must be specified with the "InputModel"
  parameter.  The number of samples to generate is specified by the "Samples"
  parameter.  Output samples may be saved with the "Output" output parameter.

  The following command can be used to generate 100 samples from the pre-trained
  GMM gmm and store those generated samples in samples:
  
  // Initialize optional parameters for GmmGenerate().
  param := mlpack.GmmGenerateOptions()
  
  samples := mlpack.GmmGenerate(&gmm, 100, param)

  Input parameters:

   - inputModel (gmm): Input GMM model to generate samples from.
   - samples (int): Number of samples to generate.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix to save output samples in.

 */
func GmmGenerate(inputModel *gmm, samples int, param *GmmGenerateOptionalParam) (*mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("GMM Sample Generator")

  // Detect if the parameter was passed; set if so.
  setGMM("input_model", inputModel)
  setPassed("input_model")

  // Detect if the parameter was passed; set if so.
  setParamInt("samples", samples)
  setPassed("samples")

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
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
  C.mlpackGmmGenerate()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat("output")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return output
}
