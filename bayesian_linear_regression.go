package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_bayesian_linear_regression
#include <capi/bayesian_linear_regression.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type BayesianLinearRegressionOptionalParam struct {
    Center bool
    Input *mat.Dense
    InputModel *bayesianLinearRegression
    Responses *mat.Dense
    Scale bool
    Test *mat.Dense
    Verbose bool
}

func BayesianLinearRegressionOptions() *BayesianLinearRegressionOptionalParam {
  return &BayesianLinearRegressionOptionalParam{
    Center: false,
    Input: nil,
    InputModel: nil,
    Responses: nil,
    Scale: false,
    Test: nil,
    Verbose: false,
  }
}

/*
  An implementation of the bayesian linear regression.
  This model is a probabilistic view and implementation of the linear
  regression. The final solution is obtained by computing a posterior
  distribution from gaussian likelihood and a zero mean gaussian isotropic 
  prior distribution on the solution. 
  Optimization is AUTOMATIC and does not require cross validation. The
  optimization is performed by maximization of the evidence function. Parameters
  are tuned during the maximization of the marginal likelihood. This procedure
  includes the Ockham's razor that penalizes over complex solutions. 
  
  This program is able to train a Bayesian linear regression model or load a
  model from file, output regression predictions for a test set, and save the
  trained model to a file.
  
  To train a BayesianLinearRegression model, the "Input" and
  "Responses"parameters must be given. The "Center"and "Scale" parameters
  control the centering and the normalizing options. A trained model can be
  saved with the "OutputModel". If no training is desired at all, a model can be
  passed via the "InputModel" parameter.
  
  The program can also provide predictions for test data using either the
  trained model or the given input model.  Test points can be specified with the
  "Test" parameter.  Predicted responses to the test points can be saved with
  the "Predictions" output parameter. The corresponding standard deviation can
  be save by precising the "Stds" parameter.

  For example, the following command trains a model on the data data and
  responses responseswith center set to true and scale set to false (so,
  Bayesian linear regression is being solved, and then the model is saved to
  blr_model:
  
  // Initialize optional parameters for BayesianLinearRegression().
  param := mlpack.BayesianLinearRegressionOptions()
  param.Input = data
  param.Responses = responses
  param.Center = 1
  param.Scale = 0
  
  blr_model, _, _ := mlpack.BayesianLinearRegression(param)
  
  The following command uses the blr_model to provide predicted  responses for
  the data test and save those  responses to test_predictions: 
  
  // Initialize optional parameters for BayesianLinearRegression().
  param := mlpack.BayesianLinearRegressionOptions()
  param.InputModel = &blr_model
  param.Test = test
  
  _, test_predictions, _ := mlpack.BayesianLinearRegression(param)
  
  Because the estimator computes a predictive distribution instead of a simple
  point estimate, the "Stds" parameter allows one to save the prediction
  uncertainties: 
  
  // Initialize optional parameters for BayesianLinearRegression().
  param := mlpack.BayesianLinearRegressionOptions()
  param.InputModel = &blr_model
  param.Test = test
  
  _, test_predictions, stds := mlpack.BayesianLinearRegression(param)

  Input parameters:

   - Center (bool): Center the data and fit the intercept if enabled.
   - Input (mat.Dense): Matrix of covariates (X).
   - InputModel (bayesianLinearRegression): Trained
        BayesianLinearRegression model to use.
   - Responses (mat.Dense): Matrix of responses/observations (y).
   - Scale (bool): Scale each feature by their standard deviations if
        enabled.
   - Test (mat.Dense): Matrix containing points to regress on (test
        points).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (bayesianLinearRegression): Output
        BayesianLinearRegression model.
   - predictions (mat.Dense): If --test_file is specified, this file is
        where the predicted responses will be saved.
   - stds (mat.Dense): If specified, this is where the standard deviations
        of the predictive distribution will be saved.

 */
func BayesianLinearRegression(param *BayesianLinearRegressionOptionalParam) (bayesianLinearRegression, *mat.Dense, *mat.Dense) {
  params := getParams("bayesian_linear_regression")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.Center != false {
    setParamBool(params, "center", param.Center)
    setPassed(params, "center")
  }

  // Detect if the parameter was passed; set if so.
  if param.Input != nil {
    gonumToArmaMat(params, "input", param.Input)
    setPassed(params, "input")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setBayesianLinearRegression(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Responses != nil {
    gonumToArmaRow(params, "responses", param.Responses)
    setPassed(params, "responses")
  }

  // Detect if the parameter was passed; set if so.
  if param.Scale != false {
    setParamBool(params, "scale", param.Scale)
    setPassed(params, "scale")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat(params, "test", param.Test)
    setPassed(params, "test")
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
  setPassed(params, "stds")

  // Call the mlpack program.
  C.mlpackBayesianLinearRegression(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputModel bayesianLinearRegression
  outputModel.getBayesianLinearRegression(params, "output_model")
  var predictionsPtr mlpackArma
  predictions := predictionsPtr.armaToGonumMat(params, "predictions")
  var stdsPtr mlpackArma
  stds := stdsPtr.armaToGonumMat(params, "stds")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return outputModel, predictions, stds
}
