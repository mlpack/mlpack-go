package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_adaboost
#include <capi/adaboost.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type AdaboostOptionalParam struct {
    InputModel *adaBoostModel
    Iterations int
    Labels *mat.Dense
    Test *mat.Dense
    Tolerance float64
    Training *mat.Dense
    Verbose bool
    WeakLearner string
}

func AdaboostOptions() *AdaboostOptionalParam {
  return &AdaboostOptionalParam{
    InputModel: nil,
    Iterations: 1000,
    Labels: nil,
    Test: nil,
    Tolerance: 1e-10,
    Training: nil,
    Verbose: false,
    WeakLearner: "decision_stump",
  }
}

/*
  This program implements the AdaBoost (or Adaptive Boosting) algorithm. The
  variant of AdaBoost implemented here is AdaBoost.MH. It uses a weak learner,
  either decision stumps or perceptrons, and over many iterations, creates a
  strong learner that is a weighted ensemble of weak learners. It runs these
  iterations until a tolerance value is crossed for change in the value of the
  weighted training error.
  
  For more information about the algorithm, see the paper "Improved Boosting
  Algorithms Using Confidence-Rated Predictions", by R.E. Schapire and Y.
  Singer.
  
  This program allows training of an AdaBoost model, and then application of
  that model to a test dataset.  To train a model, a dataset must be passed with
  the "Training" option.  Labels can be given with the "Labels" option; if no
  labels are specified, the labels will be assumed to be the last column of the
  input dataset.  Alternately, an AdaBoost model may be loaded with the
  "InputModel" option.
  
  Once a model is trained or loaded, it may be used to provide class predictions
  for a given test dataset.  A test dataset may be specified with the "Test"
  parameter.  The predicted classes for each point in the test dataset are
  output to the "Predictions" output parameter.  The AdaBoost model itself is
  output to the "OutputModel" output parameter.
  
  Note: the following parameter is deprecated and will be removed in mlpack
  4.0.0: "Output".
  Use "Predictions" instead of "Output".

  For example, to run AdaBoost on an input dataset data with labels labelsand
  perceptrons as the weak learner type, storing the trained model in model, one
  could use the following command: 
  
  // Initialize optional parameters for Adaboost().
  param := mlpack.AdaboostOptions()
  param.Training = data
  param.Labels = labels
  param.WeakLearner = "perceptron"
  
  _, model, _, _ := mlpack.Adaboost(param)
  
  Similarly, an already-trained model in model can be used to provide class
  predictions from test data test_data and store the output in predictions with
  the following command: 
  
  // Initialize optional parameters for Adaboost().
  param := mlpack.AdaboostOptions()
  param.InputModel = &model
  param.Test = test_data
  
  _, _, predictions, _ := mlpack.Adaboost(param)

  Input parameters:

   - InputModel (adaBoostModel): Input AdaBoost model.
   - Iterations (int): The maximum number of boosting iterations to be run
        (0 will run until convergence.)  Default value 1000.
   - Labels (mat.Dense): Labels for the training set.
   - Test (mat.Dense): Test dataset.
   - Tolerance (float64): The tolerance for change in values of the
        weighted error during training.  Default value 1e-10.
   - Training (mat.Dense): Dataset for training AdaBoost.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.
   - WeakLearner (string): The type of weak learner to use:
        'decision_stump', or 'perceptron'.  Default value 'decision_stump'.

  Output parameters:

   - output (mat.Dense): Predicted labels for the test set.
   - outputModel (adaBoostModel): Output trained AdaBoost model.
   - predictions (mat.Dense): Predicted labels for the test set.
   - probabilities (mat.Dense): Predicted class probabilities for each
        point in the test set.

 */
func Adaboost(param *AdaboostOptionalParam) (*mat.Dense, adaBoostModel, *mat.Dense, *mat.Dense) {
  params := getParams("adaboost")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setAdaBoostModel(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Iterations != 1000 {
    setParamInt(params, "iterations", param.Iterations)
    setPassed(params, "iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow(params, "labels", param.Labels)
    setPassed(params, "labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat(params, "test", param.Test)
    setPassed(params, "test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tolerance != 1e-10 {
    setParamDouble(params, "tolerance", param.Tolerance)
    setPassed(params, "tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    gonumToArmaMat(params, "training", param.Training)
    setPassed(params, "training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Detect if the parameter was passed; set if so.
  if param.WeakLearner != "decision_stump" {
    setParamString(params, "weak_learner", param.WeakLearner)
    setPassed(params, "weak_learner")
  }

  // Mark all output options as passed.
  setPassed(params, "output")
  setPassed(params, "output_model")
  setPassed(params, "predictions")
  setPassed(params, "probabilities")

  // Call the mlpack program.
  C.mlpackAdaboost(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumUrow(params, "output")
  var outputModel adaBoostModel
  outputModel.getAdaBoostModel(params, "output_model")
  var predictionsPtr mlpackArma
  predictions := predictionsPtr.armaToGonumUrow(params, "predictions")
  var probabilitiesPtr mlpackArma
  probabilities := probabilitiesPtr.armaToGonumMat(params, "probabilities")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return output, outputModel, predictions, probabilities
}
