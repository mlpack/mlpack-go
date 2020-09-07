package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_hmm_viterbi
#include <capi/hmm_viterbi.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type HmmViterbiOptionalParam struct {
    Verbose bool
}

func HmmViterbiOptions() *HmmViterbiOptionalParam {
  return &HmmViterbiOptionalParam{
    Verbose: false,
  }
}

/*
  This utility takes an already-trained HMM, specified as "InputModel", and
  evaluates the most probable hidden state sequence of a given sequence of
  observations (specified as '"Input", using the Viterbi algorithm.  The
  computed state sequence may be saved using the "Output" output parameter.

  For example, to predict the state sequence of the observations obs using the
  HMM hmm, storing the predicted state sequence to states, the following command
  could be used:
  
  // Initialize optional parameters for HmmViterbi().
  param := mlpack.HmmViterbiOptions()
  
  states := mlpack.HmmViterbi(obs, &hmm, param)

  Input parameters:

   - input (mat.Dense): Matrix containing observations,
   - inputModel (hmmModel): Trained HMM to use.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): File to save predicted state sequence to.

 */
func HmmViterbi(input *mat.Dense, inputModel *hmmModel, param *HmmViterbiOptionalParam) (*mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Hidden Markov Model (HMM) Viterbi State Prediction")

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
  setPassed("output")

  // Call the mlpack program.
  C.mlpackHmmViterbi()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumUmat("output")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return output
}
