package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_nbc
#include <capi/nbc.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type NbcOptionalParam struct {
    IncrementalVariance bool
    InputModel *nbcModel
    Labels *mat.Dense
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func NbcOptions() *NbcOptionalParam {
  return &NbcOptionalParam{
    IncrementalVariance: false,
    InputModel: nil,
    Labels: nil,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

/*
  This program trains the Naive Bayes classifier on the given labeled training
  set, or loads a model from the given model file, and then may use that trained
  model to classify the points in a given test set.
  
  The training set is specified with the "Training" parameter.  Labels may be
  either the last row of the training set, or alternately the "Labels" parameter
  may be specified to pass a separate matrix of labels.
  
  If training is not desired, a pre-existing model may be loaded with the
  "InputModel" parameter.
  
  
  
  The "IncrementalVariance" parameter can be used to force the training to use
  an incremental algorithm for calculating variance.  This is slower, but can
  help avoid loss of precision in some cases.
  
  If classifying a test set is desired, the test set may be specified with the
  "Test" parameter, and the classifications may be saved with the
  "Predictions"predictions  parameter.  If saving the trained model is desired,
  this may be done with the "OutputModel" output parameter.
  
  Note: the "Output" and "OutputProbs" parameters are deprecated and will be
  removed in mlpack 4.0.0.  Use "Predictions" and "Probabilities" instead.

  For example, to train a Naive Bayes classifier on the dataset data with labels
  labels and save the model to nbc_model, the following command may be used:
  
  // Initialize optional parameters for Nbc().
  param := mlpack.NbcOptions()
  param.Training = data
  param.Labels = labels
  
  _, nbc_model, _, _, _ := mlpack.Nbc(param)
  
  Then, to use nbc_model to predict the classes of the dataset test_set and save
  the predicted classes to predictions, the following command may be used:
  
  // Initialize optional parameters for Nbc().
  param := mlpack.NbcOptions()
  param.InputModel = &nbc_model
  param.Test = test_set
  
  predictions, _, _, _, _ := mlpack.Nbc(param)

  Input parameters:

   - IncrementalVariance (bool): The variance of each class will be
        calculated incrementally.
   - InputModel (nbcModel): Input Naive Bayes model.
   - Labels (mat.Dense): A file containing labels for the training set.
   - Test (mat.Dense): A matrix containing the test set.
   - Training (mat.Dense): A matrix containing the training set.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): The matrix in which the predicted labels for the
        test set will be written (deprecated).
   - outputModel (nbcModel): File to save trained Naive Bayes model to.
   - outputProbs (mat.Dense): The matrix in which the predicted
        probability of labels for the test set will be written (deprecated).
   - predictions (mat.Dense): The matrix in which the predicted labels for
        the test set will be written.
   - probabilities (mat.Dense): The matrix in which the predicted
        probability of labels for the test set will be written.

 */
func Nbc(param *NbcOptionalParam) (*mat.Dense, nbcModel, *mat.Dense, *mat.Dense, *mat.Dense) {
  params := getParams("nbc")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.IncrementalVariance != false {
    setParamBool(params, "incremental_variance", param.IncrementalVariance)
    setPassed(params, "incremental_variance")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setNBCModel(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow(params, "labels", param.Labels)
    setPassed(params, "labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat(params, "test", param.Test, false)
    setPassed(params, "test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    gonumToArmaMat(params, "training", param.Training, false)
    setPassed(params, "training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "output")
  setPassed(params, "output_model")
  setPassed(params, "output_probs")
  setPassed(params, "predictions")
  setPassed(params, "probabilities")

  // Call the mlpack program.
  C.mlpackNbc(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumUrow(params, "output")
  var outputModel nbcModel
  outputModel.getNBCModel(params, "output_model")
  var outputProbsPtr mlpackArma
  outputProbs := outputProbsPtr.armaToGonumMat(params, "output_probs")
  var predictionsPtr mlpackArma
  predictions := predictionsPtr.armaToGonumUrow(params, "predictions")
  var probabilitiesPtr mlpackArma
  probabilities := probabilitiesPtr.armaToGonumMat(params, "probabilities")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return output, outputModel, outputProbs, predictions, probabilities
}
