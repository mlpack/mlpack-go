package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_lars
#include <capi/lars.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type LarsOptionalParam struct {
    Input *mat.Dense
    InputModel *lars
    Lambda1 float64
    Lambda2 float64
    Responses *mat.Dense
    Test *mat.Dense
    UseCholesky bool
    Verbose bool
}

func LarsOptions() *LarsOptionalParam {
  return &LarsOptionalParam{
    Input: nil,
    InputModel: nil,
    Lambda1: 0,
    Lambda2: 0,
    Responses: nil,
    Test: nil,
    UseCholesky: false,
    Verbose: false,
  }
}

/*
  An implementation of LARS: Least Angle Regression (Stagewise/laSso).  This is
  a stage-wise homotopy-based algorithm for L1-regularized linear regression
  (LASSO) and L1+L2-regularized linear regression (Elastic Net).
  
  This program is able to train a LARS/LASSO/Elastic Net model or load a model
  from file, output regression predictions for a test set, and save the trained
  model to a file.  The LARS algorithm is described in more detail below:
  
  Let X be a matrix where each row is a point and each column is a dimension,
  and let y be a vector of targets.
  
  The Elastic Net problem is to solve
  
    min_beta 0.5 || X * beta - y ||_2^2 + lambda_1 ||beta||_1 +
        0.5 lambda_2 ||beta||_2^2
  
  If lambda1 > 0 and lambda2 = 0, the problem is the LASSO.
  If lambda1 > 0 and lambda2 > 0, the problem is the Elastic Net.
  If lambda1 = 0 and lambda2 > 0, the problem is ridge regression.
  If lambda1 = 0 and lambda2 = 0, the problem is unregularized linear
  regression.
  
  For efficiency reasons, it is not recommended to use this algorithm with
  "Lambda1" = 0.  In that case, use the 'linear_regression' program, which
  implements both unregularized linear regression and ridge regression.
  
  To train a LARS/LASSO/Elastic Net model, the "Input" and "Responses"
  parameters must be given.  The "Lambda1", "Lambda2", and "UseCholesky"
  parameters control the training options.  A trained model can be saved with
  the "OutputModel".  If no training is desired at all, a model can be passed
  via the "InputModel" parameter.
  
  The program can also provide predictions for test data using either the
  trained model or the given input model.  Test points can be specified with the
  "Test" parameter.  Predicted responses to the test points can be saved with
  the "OutputPredictions" output parameter.

  For example, the following command trains a model on the data data and
  responses responses with lambda1 set to 0.4 and lambda2 set to 0 (so, LASSO is
  being solved), and then the model is saved to lasso_model:
  
  // Initialize optional parameters for Lars().
  param := mlpack.LarsOptions()
  param.Input = data
  param.Responses = responses
  param.Lambda1 = 0.4
  param.Lambda2 = 0
  
  lasso_model, _ := mlpack.Lars(param)
  
  The following command uses the lasso_model to provide predicted responses for
  the data test and save those responses to test_predictions: 
  
  // Initialize optional parameters for Lars().
  param := mlpack.LarsOptions()
  param.InputModel = &lasso_model
  param.Test = test
  
  _, test_predictions := mlpack.Lars(param)

  Input parameters:

   - Input (mat.Dense): Matrix of covariates (X).
   - InputModel (lars): Trained LARS model to use.
   - Lambda1 (float64): Regularization parameter for l1-norm penalty. 
        Default value 0.
   - Lambda2 (float64): Regularization parameter for l2-norm penalty. 
        Default value 0.
   - Responses (mat.Dense): Matrix of responses/observations (y).
   - Test (mat.Dense): Matrix containing points to regress on (test
        points).
   - UseCholesky (bool): Use Cholesky decomposition during computation
        rather than explicitly computing the full Gram matrix.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (lars): Output LARS model.
   - outputPredictions (mat.Dense): If --test_file is specified, this file
        is where the predicted responses will be saved.

 */
func Lars(param *LarsOptionalParam) (lars, *mat.Dense) {
  params := getParams("lars")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.Input != nil {
    gonumToArmaMat(params, "input", param.Input)
    setPassed(params, "input")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setLARS(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda1 != 0 {
    setParamDouble(params, "lambda1", param.Lambda1)
    setPassed(params, "lambda1")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda2 != 0 {
    setParamDouble(params, "lambda2", param.Lambda2)
    setPassed(params, "lambda2")
  }

  // Detect if the parameter was passed; set if so.
  if param.Responses != nil {
    gonumToArmaMat(params, "responses", param.Responses)
    setPassed(params, "responses")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat(params, "test", param.Test)
    setPassed(params, "test")
  }

  // Detect if the parameter was passed; set if so.
  if param.UseCholesky != false {
    setParamBool(params, "use_cholesky", param.UseCholesky)
    setPassed(params, "use_cholesky")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "output_model")
  setPassed(params, "output_predictions")

  // Call the mlpack program.
  C.mlpackLars(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputModel lars
  outputModel.getLARS(params, "output_model")
  var outputPredictionsPtr mlpackArma
  outputPredictions := outputPredictionsPtr.armaToGonumMat(params, "output_predictions")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return outputModel, outputPredictions
}
