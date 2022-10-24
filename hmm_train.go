package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_hmm_train
#include <capi/hmm_train.h>
#include <stdlib.h>
*/
import "C" 


type HmmTrainOptionalParam struct {
    Batch bool
    Gaussians int
    InputModel *hmmModel
    LabelsFile string
    Seed int
    States int
    Tolerance float64
    Type string
    Verbose bool
}

func HmmTrainOptions() *HmmTrainOptionalParam {
  return &HmmTrainOptionalParam{
    Batch: false,
    Gaussians: 0,
    InputModel: nil,
    LabelsFile: "",
    Seed: 0,
    States: 0,
    Tolerance: 1e-05,
    Type: "gaussian",
    Verbose: false,
  }
}

/*
  This program allows a Hidden Markov Model to be trained on labeled or
  unlabeled data.  It supports four types of HMMs: Discrete HMMs, Gaussian HMMs,
  GMM HMMs, or Diagonal GMM HMMs
  
  Either one input sequence can be specified (with "InputFile"), or, a file
  containing files in which input sequences can be found (when
  "InputFile"and"Batch" are used together).  In addition, labels can be provided
  in the file specified by "LabelsFile", and if "Batch" is used, the file given
  to "LabelsFile" should contain a list of files of labels corresponding to the
  sequences in the file given to "InputFile".
  
  The HMM is trained with the Baum-Welch algorithm if no labels are provided. 
  The tolerance of the Baum-Welch algorithm can be set with the
  "Tolerance"option.  By default, the transition matrix is randomly initialized
  and the emission distributions are initialized to fit the extent of the data.
  
  Optionally, a pre-created HMM model can be used as a guess for the transition
  matrix and emission probabilities; this is specifiable with "OutputModel".

  Input parameters:

   - inputFile (string): File containing input observations.
   - Batch (bool): If true, input_file (and if passed, labels_file) are
        expected to contain a list of files to use as input observation
        sequences (and label sequences).
   - Gaussians (int): Number of gaussians in each GMM (necessary when type
        is 'gmm').  Default value 0.
   - InputModel (hmmModel): Pre-existing HMM model to initialize training
        with.
   - LabelsFile (string): Optional file of hidden states, used for labeled
        training.  Default value ''.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - States (int): Number of hidden states in HMM (necessary, unless
        model_file is specified).  Default value 0.
   - Tolerance (float64): Tolerance of the Baum-Welch algorithm.  Default
        value 1e-05.
   - Type (string): Type of HMM: discrete | gaussian | diag_gmm | gmm. 
        Default value 'gaussian'.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (hmmModel): Output for trained HMM.

 */
func HmmTrain(inputFile string, param *HmmTrainOptionalParam) (hmmModel) {
  params := getParams("hmm_train")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  setParamString(params, "input_file", inputFile)
  setPassed(params, "input_file")

  // Detect if the parameter was passed; set if so.
  if param.Batch != false {
    setParamBool(params, "batch", param.Batch)
    setPassed(params, "batch")
  }

  // Detect if the parameter was passed; set if so.
  if param.Gaussians != 0 {
    setParamInt(params, "gaussians", param.Gaussians)
    setPassed(params, "gaussians")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setHMMModel(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.LabelsFile != "" {
    setParamString(params, "labels_file", param.LabelsFile)
    setPassed(params, "labels_file")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt(params, "seed", param.Seed)
    setPassed(params, "seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.States != 0 {
    setParamInt(params, "states", param.States)
    setPassed(params, "states")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tolerance != 1e-05 {
    setParamDouble(params, "tolerance", param.Tolerance)
    setPassed(params, "tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Type != "gaussian" {
    setParamString(params, "type", param.Type)
    setPassed(params, "type")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "output_model")

  // Call the mlpack program.
  C.mlpackHmmTrain(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputModel hmmModel
  outputModel.getHMMModel(params, "output_model")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return outputModel
}
