package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_softmax_regression
#include <capi/softmax_regression.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type SoftmaxRegressionOptionalParam struct {
    InputModel *softmaxRegression
    Labels *mat.Dense
    Lambda float64
    MaxIterations int
    NoIntercept bool
    NumberOfClasses int
    Test *mat.Dense
    TestLabels *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func SoftmaxRegressionOptions() *SoftmaxRegressionOptionalParam {
  return &SoftmaxRegressionOptionalParam{
    InputModel: nil,
    Labels: nil,
    Lambda: 0.0001,
    MaxIterations: 400,
    NoIntercept: false,
    NumberOfClasses: 0,
    Test: nil,
    TestLabels: nil,
    Training: nil,
    Verbose: false,
  }
}

/*
  This program performs softmax regression, a generalization of logistic
  regression to the multiclass case, and has support for L2 regularization.  The
  program is able to train a model, load  an existing model, and give
  predictions (and optionally their accuracy) for test data.
  
  Training a softmax regression model is done by giving a file of training
  points with the "Training" parameter and their corresponding labels with the
  "Labels" parameter. The number of classes can be manually specified with the
  "NumberOfClasses" parameter, and the maximum number of iterations of the
  L-BFGS optimizer can be specified with the "MaxIterations" parameter.  The L2
  regularization constant can be specified with the "Lambda" parameter and if an
  intercept term is not desired in the model, the "NoIntercept" parameter can be
  specified.
  
  The trained model can be saved with the "OutputModel" output parameter. If
  training is not desired, but only testing is, a model can be loaded with the
  "InputModel" parameter.  At the current time, a loaded model cannot be trained
  further, so specifying both "InputModel" and "Training" is not allowed.
  
  The program is also able to evaluate a model on test data.  A test dataset can
  be specified with the "Test" parameter. Class predictions can be saved with
  the "Predictions" output parameter.  If labels are specified for the test data
  with the "TestLabels" parameter, then the program will print the accuracy of
  the predictions on the given test set and its corresponding labels.

  For example, to train a softmax regression model on the data dataset with
  labels labels with a maximum of 1000 iterations for training, saving the
  trained model to sr_model, the following command can be used: 
  
  // Initialize optional parameters for SoftmaxRegression().
  param := mlpack.SoftmaxRegressionOptions()
  param.Training = dataset
  param.Labels = labels
  
  sr_model, _, _ := mlpack.SoftmaxRegression(param)
  
  Then, to use sr_model to classify the test points in test_points, saving the
  output predictions to predictions, the following command can be used:
  
  // Initialize optional parameters for SoftmaxRegression().
  param := mlpack.SoftmaxRegressionOptions()
  param.InputModel = &sr_model
  param.Test = test_points
  
  _, predictions, _ := mlpack.SoftmaxRegression(param)

  Input parameters:

   - InputModel (softmaxRegression): File containing existing model
        (parameters).
   - Labels (mat.Dense): A matrix containing labels (0 or 1) for the
        points in the training set (y). The labels must order as a row.
   - Lambda (float64): L2-regularization constant  Default value 0.0001.
   - MaxIterations (int): Maximum number of iterations before termination.
         Default value 400.
   - NoIntercept (bool): Do not add the intercept term to the model.
   - NumberOfClasses (int): Number of classes for classification; if
        unspecified (or 0), the number of classes found in the labels will be
        used.  Default value 0.
   - Test (mat.Dense): Matrix containing test dataset.
   - TestLabels (mat.Dense): Matrix containing test labels.
   - Training (mat.Dense): A matrix containing the training set (the
        matrix of predictors, X).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (softmaxRegression): File to save trained softmax
        regression model to.
   - predictions (mat.Dense): Matrix to save predictions for test dataset
        into.
   - probabilities (mat.Dense): Matrix to save class probabilities for
        test dataset into.

 */
func SoftmaxRegression(param *SoftmaxRegressionOptionalParam) (softmaxRegression, *mat.Dense, *mat.Dense) {
  params := getParams("softmax_regression")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setSoftmaxRegression(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow(params, "labels", param.Labels)
    setPassed(params, "labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda != 0.0001 {
    setParamDouble(params, "lambda", param.Lambda)
    setPassed(params, "lambda")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 400 {
    setParamInt(params, "max_iterations", param.MaxIterations)
    setPassed(params, "max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.NoIntercept != false {
    setParamBool(params, "no_intercept", param.NoIntercept)
    setPassed(params, "no_intercept")
  }

  // Detect if the parameter was passed; set if so.
  if param.NumberOfClasses != 0 {
    setParamInt(params, "number_of_classes", param.NumberOfClasses)
    setPassed(params, "number_of_classes")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat(params, "test", param.Test)
    setPassed(params, "test")
  }

  // Detect if the parameter was passed; set if so.
  if param.TestLabels != nil {
    gonumToArmaUrow(params, "test_labels", param.TestLabels)
    setPassed(params, "test_labels")
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

  // Mark all output options as passed.
  setPassed(params, "output_model")
  setPassed(params, "predictions")
  setPassed(params, "probabilities")

  // Call the mlpack program.
  C.mlpackSoftmaxRegression(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputModel softmaxRegression
  outputModel.getSoftmaxRegression(params, "output_model")
  var predictionsPtr mlpackArma
  predictions := predictionsPtr.armaToGonumUrow(params, "predictions")
  var probabilitiesPtr mlpackArma
  probabilities := probabilitiesPtr.armaToGonumMat(params, "probabilities")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return outputModel, predictions, probabilities
}
