package mlpack

/*
#cgo CFLAGS: -I./capi
#cgo LDFLAGS: -L. -lmlpack_go_hmm_generate
#include <capi/hmm_generate.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type HmmGenerateOptionalParam struct {
    Seed int
    StartState int
    Verbose bool
}

func HmmGenerateOptions() *HmmGenerateOptionalParam {
  return &HmmGenerateOptionalParam{
    Seed: 0,
    StartState: 0,
    Verbose: false,
  }
}

/*
  This utility takes an already-trained HMM, specified as the "Model" parameter,
  and generates a random observation sequence and hidden state sequence based on
  its parameters. The observation sequence may be saved with the "Output" output
  parameter, and the internal state  sequence may be saved with the "State"
  output parameter.
  
  The state to start the sequence in may be specified with the "StartState"
  parameter.

  For example, to generate a sequence of length 150 from the HMM hmm and save
  the observation sequence to observations and the hidden state sequence to
  states, the following command may be used: 
  
  // Initialize optional parameters for HmmGenerate().
  param := mlpack.HmmGenerateOptions()
  
  observations, states := mlpack.HmmGenerate(&hmm, 150, param)

  Input parameters:

   - length (int): Length of sequence to generate.
   - model (hmmModel): Trained HMM to generate sequences with.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - StartState (int): Starting state of sequence.  Default value 0.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix to save observation sequence to.
   - state (mat.Dense): Matrix to save hidden state sequence to.

 */
func HmmGenerate(length int, model *hmmModel, param *HmmGenerateOptionalParam) (*mat.Dense, *mat.Dense) {
  params := getParams("hmm_generate")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  setParamInt(params, "length", length)
  setPassed(params, "length")

  // Detect if the parameter was passed; set if so.
  setHMMModel(params, "model", model)
  setPassed(params, "model")

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt(params, "seed", param.Seed)
    setPassed(params, "seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.StartState != 0 {
    setParamInt(params, "start_state", param.StartState)
    setPassed(params, "start_state")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "output")
  setPassed(params, "state")

  // Call the mlpack program.
  C.mlpackHmmGenerate(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat(params, "output")
  var statePtr mlpackArma
  state := statePtr.armaToGonumUmat(params, "state")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return output, state
}
