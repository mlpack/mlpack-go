package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_linear_regression
#include <capi/linear_regression.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type LinearRegressionOptionalParam struct {
    InputModel *linearRegression
    Lambda float64
    Test *mat.Dense
    Training *mat.Dense
    TrainingResponses *mat.Dense
    Verbose bool
}

func LinearRegressionOptions() *LinearRegressionOptionalParam {
  return &LinearRegressionOptionalParam{
    InputModel: nil,
    Lambda: 0,
    Test: nil,
    Training: nil,
    TrainingResponses: nil,
    Verbose: false,
  }
}

type linearRegression struct {
  mem unsafe.Pointer
}

func (m *linearRegression) allocLinearRegression(identifier string) {
  m.mem = C.mlpackGetLinearRegressionPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *linearRegression) getLinearRegression(identifier string) {
  m.allocLinearRegression(identifier)
}

func setLinearRegression(identifier string, ptr *linearRegression) {
  C.mlpackSetLinearRegressionPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  An implementation of simple linear regression and simple ridge regression
  using ordinary least squares. This solves the problem
  
    y = X * b + e
  
  where X (specified by "Training") and y (specified either as the last column
  of the input matrix "Training" or via the "TrainingResponses" parameter) are
  known and b is the desired variable.  If the covariance matrix (X'X) is not
  invertible, or if the solution is overdetermined, then specify a Tikhonov
  regularization constant (with "Lambda") greater than 0, which will regularize
  the covariance matrix to make it invertible.  The calculated b may be saved
  with the "OutputPredictions" output parameter.
  
  Optionally, the calculated value of b is used to predict the responses for
  another matrix X' (specified by the "Test" parameter):
  
     y' = X' * b
  
  and the predicted responses y' may be saved with the "OutputPredictions"
  output parameter.  This type of regression is related to least-angle
  regression, which mlpack implements as the 'lars' program.
  
  For example, to run a linear regression on the dataset X with responses y,
  saving the trained model to lr_model, the following command could be used:
  
  // Initialize optional parameters for LinearRegression().
  param := mlpack.LinearRegressionOptions()
  param.Training = X
  param.TrainingResponses = y
  
  lr_model, _ := mlpack.LinearRegression(param)
  
  Then, to use lr_model to predict responses for a test set X_test, saving the
  predictions to X_test_responses, the following command could be used:
  
  // Initialize optional parameters for LinearRegression().
  param := mlpack.LinearRegressionOptions()
  param.InputModel = &lr_model
  param.Test = X_test
  
  _, X_test_responses := mlpack.LinearRegression(param)


  Input parameters:

   - InputModel (linearRegression): Existing LinearRegression model to
        use.
   - Lambda (float64): Tikhonov regularization for ridge regression.  If
        0, the method reduces to linear regression.  Default value 0.
   - Test (mat.Dense): Matrix containing X' (test regressors).
   - Training (mat.Dense): Matrix containing training set X (regressors).
   - TrainingResponses (mat.Dense): Optional vector containing y
        (responses). If not given, the responses are assumed to be the last row
        of the input file.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (linearRegression): Output LinearRegression model.
   - outputPredictions (mat.Dense): If --test_file is specified, this
        matrix is where the predicted responses will be saved.

 */
func LinearRegression(param *LinearRegressionOptionalParam) (linearRegression, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Simple Linear Regression and Prediction")

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setLinearRegression("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda != 0 {
    setParamDouble("lambda", param.Lambda)
    setPassed("lambda")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat("test", param.Test)
    setPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    gonumToArmaMat("training", param.Training)
    setPassed("training")
  }

  // Detect if the parameter was passed; set if so.
  if param.TrainingResponses != nil {
    gonumToArmaRow("training_responses", param.TrainingResponses)
    setPassed("training_responses")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("output_model")
  setPassed("output_predictions")

  // Call the mlpack program.
  C.mlpackLinearRegression()

  // Initialize result variable and get output.
  var outputModel linearRegression
  outputModel.getLinearRegression("output_model")
  var outputPredictionsPtr mlpackArma
  outputPredictions := outputPredictionsPtr.armaToGonumRow("output_predictions")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return outputModel, outputPredictions
}
