package mlpack

/*
#cgo CFLAGS: -I./capi
#cgo LDFLAGS: -L. -lmlpack_go_gmm_probability
#include <capi/gmm_probability.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type GmmProbabilityOptionalParam struct {
    Verbose bool
}

func GmmProbabilityOptions() *GmmProbabilityOptionalParam {
  return &GmmProbabilityOptionalParam{
    Verbose: false,
  }
}

/*
  This program calculates the probability that given points came from a given
  GMM (that is, P(X | gmm)).  The GMM is specified with the "InputModel"
  parameter, and the points are specified with the "Input" parameter.  The
  output probabilities may be saved via the "Output" output parameter.

  So, for example, to calculate the probabilities of each point in points coming
  from the pre-trained GMM gmm, while storing those probabilities in probs, the
  following command could be used:
  
  // Initialize optional parameters for GmmProbability().
  param := mlpack.GmmProbabilityOptions()
  
  probs := mlpack.GmmProbability(&gmm, points, param)

  Input parameters:

   - input (mat.Dense): Input matrix to calculate probabilities of.
   - inputModel (gmm): Input GMM to use as model.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix to store calculated probabilities in.

 */
func GmmProbability(input *mat.Dense, inputModel *gmm, param *GmmProbabilityOptionalParam) (*mat.Dense) {
  params := getParams("gmm_probability")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  gonumToArmaMat(params, "input", input, false)
  setPassed(params, "input")

  // Detect if the parameter was passed; set if so.
  setGMM(params, "input_model", inputModel)
  setPassed(params, "input_model")

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "output")

  // Call the mlpack program.
  C.mlpackGmmProbability(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat(params, "output")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return output
}
