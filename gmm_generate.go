package mlpack

/*
#cgo CFLAGS: -I./capi
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
  params := getParams("gmm_generate")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  setGMM(params, "input_model", inputModel)
  setPassed(params, "input_model")

  // Detect if the parameter was passed; set if so.
  setParamInt(params, "samples", samples)
  setPassed(params, "samples")

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt(params, "seed", param.Seed)
    setPassed(params, "seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "output")

  // Call the mlpack program.
  C.mlpackGmmGenerate(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat(params, "output")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return output
}
