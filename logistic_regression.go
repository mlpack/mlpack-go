package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_logistic_regression
#include <capi/logistic_regression.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type LogisticRegressionOptionalParam struct {
    BatchSize int
    DecisionBoundary float64
    InputModel *logisticRegression
    Labels *mat.Dense
    Lambda float64
    MaxIterations int
    Optimizer string
    StepSize float64
    Test *mat.Dense
    Tolerance float64
    Training *mat.Dense
    Verbose bool
}

func LogisticRegressionOptions() *LogisticRegressionOptionalParam {
  return &LogisticRegressionOptionalParam{
    BatchSize: 64,
    DecisionBoundary: 0.5,
    InputModel: nil,
    Labels: nil,
    Lambda: 0,
    MaxIterations: 10000,
    Optimizer: "lbfgs",
    StepSize: 0.01,
    Test: nil,
    Tolerance: 1e-10,
    Training: nil,
    Verbose: false,
  }
}

/*
  An implementation of L2-regularized logistic regression using either the
  L-BFGS optimizer or SGD (stochastic gradient descent).  This solves the
  regression problem
  
    y = (1 / 1 + e^-(X * b))
  
  where y takes values 0 or 1.
  
  This program allows loading a logistic regression model (via the "InputModel"
  parameter) or training a logistic regression model given training data
  (specified with the "Training" parameter), or both those things at once.  In
  addition, this program allows classification on a test dataset (specified with
  the "Test" parameter) and the classification results may be saved with the
  "Predictions" output parameter. The trained logistic regression model may be
  saved using the "OutputModel" output parameter.
  
  The training data, if specified, may have class labels as its last dimension. 
  Alternately, the "Labels" parameter may be used to specify a separate matrix
  of labels.
  
  When a model is being trained, there are many options.  L2 regularization (to
  prevent overfitting) can be specified with the "Lambda" option, and the
  optimizer used to train the model can be specified with the "Optimizer"
  parameter.  Available options are 'sgd' (stochastic gradient descent) and
  'lbfgs' (the L-BFGS optimizer).  There are also various parameters for the
  optimizer; the "MaxIterations" parameter specifies the maximum number of
  allowed iterations, and the "Tolerance" parameter specifies the tolerance for
  convergence.  For the SGD optimizer, the "StepSize" parameter controls the
  step size taken at each iteration by the optimizer.  The batch size for SGD is
  controlled with the "BatchSize" parameter. If the objective function for your
  data is oscillating between Inf and 0, the step size is probably too large. 
  There are more parameters for the optimizers, but the C++ interface must be
  used to access these.
  
  For SGD, an iteration refers to a single point. So to take a single pass over
  the dataset with SGD, "MaxIterations" should be set to the number of points in
  the dataset.
  
  Optionally, the model can be used to predict the responses for another matrix
  of data points, if "Test" is specified.  The "Test" parameter can be specified
  without the "Training" parameter, so long as an existing logistic regression
  model is given with the "InputModel" parameter.  The output predictions from
  the logistic regression model may be saved with the "Predictions" parameter.
  
  Note : The following parameters are deprecated and will be removed in mlpack
  4: "Output", "OutputProbabilities"
  Use "Predictions" instead of "Output"
  Use "Probabilities" instead of "OutputProbabilities"
  
  This implementation of logistic regression does not support the general
  multi-class case but instead only the two-class case.  Any labels must be
  either 0 or 1.  For more classes, see the softmax_regression program.

  As an example, to train a logistic regression model on the data 'data' with
  labels 'labels' with L2 regularization of 0.1, saving the model to 'lr_model',
  the following command may be used:
  
  // Initialize optional parameters for LogisticRegression().
  param := mlpack.LogisticRegressionOptions()
  param.Training = data
  param.Labels = labels
  param.Lambda = 0.1
  
  _, lr_model, _, _, _ := mlpack.LogisticRegression(param)
  
  Then, to use that model to predict classes for the dataset 'test', storing the
  output predictions in 'predictions', the following command may be used: 
  
  // Initialize optional parameters for LogisticRegression().
  param := mlpack.LogisticRegressionOptions()
  param.InputModel = &lr_model
  param.Test = test
  
  predictions, _, _, _, _ := mlpack.LogisticRegression(param)

  Input parameters:

   - BatchSize (int): Batch size for SGD.  Default value 64.
   - DecisionBoundary (float64): Decision boundary for prediction; if the
        logistic function for a point is less than the boundary, the class is
        taken to be 0; otherwise, the class is 1.  Default value 0.5.
   - InputModel (logisticRegression): Existing model (parameters).
   - Labels (mat.Dense): A matrix containing labels (0 or 1) for the
        points in the training set (y).
   - Lambda (float64): L2-regularization parameter for training.  Default
        value 0.
   - MaxIterations (int): Maximum iterations for optimizer (0 indicates no
        limit).  Default value 10000.
   - Optimizer (string): Optimizer to use for training ('lbfgs' or 'sgd').
         Default value 'lbfgs'.
   - StepSize (float64): Step size for SGD optimizer.  Default value
        0.01.
   - Test (mat.Dense): Matrix containing test dataset.
   - Tolerance (float64): Convergence tolerance for optimizer.  Default
        value 1e-10.
   - Training (mat.Dense): A matrix containing the training set (the
        matrix of predictors, X).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): If test data is specified, this matrix is where
        the predictions for the test set will be saved.
   - outputModel (logisticRegression): Output for trained logistic
        regression model.
   - outputProbabilities (mat.Dense): If test data is specified, this
        matrix is where the class probabilities for the test set will be saved.
   - predictions (mat.Dense): If test data is specified, this matrix is
        where the predictions for the test set will be saved.
   - probabilities (mat.Dense): If test data is specified, this matrix is
        where the class probabilities for the test set will be saved.

 */
func LogisticRegression(param *LogisticRegressionOptionalParam) (*mat.Dense, logisticRegression, *mat.Dense, *mat.Dense, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("L2-regularized Logistic Regression and Prediction")

  // Detect if the parameter was passed; set if so.
  if param.BatchSize != 64 {
    setParamInt("batch_size", param.BatchSize)
    setPassed("batch_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.DecisionBoundary != 0.5 {
    setParamDouble("decision_boundary", param.DecisionBoundary)
    setPassed("decision_boundary")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setLogisticRegression("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow("labels", param.Labels)
    setPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda != 0 {
    setParamDouble("lambda", param.Lambda)
    setPassed("lambda")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 10000 {
    setParamInt("max_iterations", param.MaxIterations)
    setPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Optimizer != "lbfgs" {
    setParamString("optimizer", param.Optimizer)
    setPassed("optimizer")
  }

  // Detect if the parameter was passed; set if so.
  if param.StepSize != 0.01 {
    setParamDouble("step_size", param.StepSize)
    setPassed("step_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat("test", param.Test)
    setPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tolerance != 1e-10 {
    setParamDouble("tolerance", param.Tolerance)
    setPassed("tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    gonumToArmaMat("training", param.Training)
    setPassed("training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("output")
  setPassed("output_model")
  setPassed("output_probabilities")
  setPassed("predictions")
  setPassed("probabilities")

  // Call the mlpack program.
  C.mlpackLogisticRegression()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumUrow("output")
  var outputModel logisticRegression
  outputModel.getLogisticRegression("output_model")
  var outputProbabilitiesPtr mlpackArma
  outputProbabilities := outputProbabilitiesPtr.armaToGonumMat("output_probabilities")
  var predictionsPtr mlpackArma
  predictions := predictionsPtr.armaToGonumUrow("predictions")
  var probabilitiesPtr mlpackArma
  probabilities := probabilitiesPtr.armaToGonumMat("probabilities")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return output, outputModel, outputProbabilities, predictions, probabilities
}
