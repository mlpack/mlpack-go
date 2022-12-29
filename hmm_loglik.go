package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_hmm_loglik
#include <capi/hmm_loglik.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type HmmLoglikOptionalParam struct {
    Verbose bool
}

func HmmLoglikOptions() *HmmLoglikOptionalParam {
  return &HmmLoglikOptionalParam{
    Verbose: false,
  }
}

/*
  This utility takes an already-trained HMM, specified with the "InputModel"
  parameter, and evaluates the log-likelihood of a sequence of observations,
  given with the "Input" parameter.  The computed log-likelihood is given as
  output.

  For example, to compute the log-likelihood of the sequence seq with the
  pre-trained HMM hmm, the following command may be used: 
  
  // Initialize optional parameters for HmmLoglik().
  param := mlpack.HmmLoglikOptions()
  
  _ := mlpack.HmmLoglik(seq, &hmm, param)

  Input parameters:

   - input (mat.Dense): File containing observations,
   - inputModel (hmmModel): File containing HMM.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - logLikelihood (float64): Log-likelihood of the sequence.  Default
        value 0.

 */
func HmmLoglik(input *mat.Dense, inputModel *hmmModel, param *HmmLoglikOptionalParam) (float64) {
  params := getParams("hmm_loglik")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  gonumToArmaMat(params, "input", input, false)
  setPassed(params, "input")

  // Detect if the parameter was passed; set if so.
  setHMMModel(params, "input_model", inputModel)
  setPassed(params, "input_model")

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "log_likelihood")

  // Call the mlpack program.
  C.mlpackHmmLoglik(params.mem, timers.mem)

  // Initialize result variable and get output.
  logLikelihood := getParamDouble(params, "log_likelihood")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return logLikelihood
}
