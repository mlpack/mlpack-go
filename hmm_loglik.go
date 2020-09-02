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
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Hidden Markov Model (HMM) Sequence Log-Likelihood")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  setHMMModel("input_model", inputModel)
  setPassed("input_model")

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("log_likelihood")

  // Call the mlpack program.
  C.mlpackHmmLoglik()

  // Initialize result variable and get output.
  logLikelihood := getParamDouble("log_likelihood")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return logLikelihood
}
