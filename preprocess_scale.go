package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_preprocess_scale
#include <capi/preprocess_scale.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type PreprocessScaleOptionalParam struct {
    Epsilon float64
    InputModel *scalingModel
    InverseScaling bool
    MaxValue int
    MinValue int
    ScalerMethod string
    Seed int
    Verbose bool
}

func PreprocessScaleOptions() *PreprocessScaleOptionalParam {
  return &PreprocessScaleOptionalParam{
    Epsilon: 1e-06,
    InputModel: nil,
    InverseScaling: false,
    MaxValue: 1,
    MinValue: 0,
    ScalerMethod: "standard_scaler",
    Seed: 0,
    Verbose: false,
  }
}

/*
  This utility takes a dataset and performs feature scaling using one of the six
  scaler methods namely: 'max_abs_scaler', 'mean_normalization',
  'min_max_scaler' ,'standard_scaler', 'pca_whitening' and 'zca_whitening'. The
  function takes a matrix as "Input" and a scaling method type which you can
  specify using "ScalerMethod" parameter; the default is standard scaler, and
  outputs a matrix with scaled feature.
  
  The output scaled feature matrix may be saved with the "Output" output
  parameters.
  
  The model to scale features can be saved using "OutputModel" and later can be
  loaded back using"InputModel".

  So, a simple example where we want to scale the dataset X into X_scaled with 
  standard_scaler as scaler_method, we could run 
  
  // Initialize optional parameters for PreprocessScale().
  param := mlpack.PreprocessScaleOptions()
  param.ScalerMethod = "standard_scaler"
  
  X_scaled, _ := mlpack.PreprocessScale(X, param)
  
  A simple example where we want to whiten the dataset X into X_whitened with 
  PCA as whitening_method and use 0.01 as regularization parameter, we could run
  
  
  // Initialize optional parameters for PreprocessScale().
  param := mlpack.PreprocessScaleOptions()
  param.ScalerMethod = "pca_whitening"
  param.Epsilon = 0.01
  
  X_scaled, _ := mlpack.PreprocessScale(X, param)
  
  You can also retransform the scaled dataset back using"InverseScaling". An
  example to rescale : X_scaled into Xusing the saved model "InputModel" is:
  
  // Initialize optional parameters for PreprocessScale().
  param := mlpack.PreprocessScaleOptions()
  param.InverseScaling = true
  param.InputModel = &saved
  
  X, _ := mlpack.PreprocessScale(X_scaled, param)
  
  Another simple example where we want to scale the dataset X into X_scaled with
   min_max_scaler as scaler method, where scaling range is 1 to 3 instead of
  default 0 to 1. We could run 
  
  // Initialize optional parameters for PreprocessScale().
  param := mlpack.PreprocessScaleOptions()
  param.ScalerMethod = "min_max_scaler"
  param.MinValue = 1
  param.MaxValue = 3
  
  X_scaled, _ := mlpack.PreprocessScale(X, param)

  Input parameters:

   - input (mat.Dense): Matrix containing data.
   - Epsilon (float64): regularization Parameter for pcawhitening, or
        zcawhitening, should be between -1 to 1.  Default value 1e-06.
   - InputModel (scalingModel): Input Scaling model.
   - InverseScaling (bool): Inverse Scaling to get original dataset
   - MaxValue (int): Ending value of range for min_max_scaler.  Default
        value 1.
   - MinValue (int): Starting value of range for min_max_scaler.  Default
        value 0.
   - ScalerMethod (string): method to use for scaling, the default is
        standard_scaler.  Default value 'standard_scaler'.
   - Seed (int): Random seed (0 for std::time(NULL)).  Default value 0.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix to save scaled data to.
   - outputModel (scalingModel): Output scaling model.

 */
func PreprocessScale(input *mat.Dense, param *PreprocessScaleOptionalParam) (*mat.Dense, scalingModel) {
  params := getParams("preprocess_scale")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  gonumToArmaMat(params, "input", input, false)
  setPassed(params, "input")

  // Detect if the parameter was passed; set if so.
  if param.Epsilon != 1e-06 {
    setParamDouble(params, "epsilon", param.Epsilon)
    setPassed(params, "epsilon")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setScalingModel(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.InverseScaling != false {
    setParamBool(params, "inverse_scaling", param.InverseScaling)
    setPassed(params, "inverse_scaling")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxValue != 1 {
    setParamInt(params, "max_value", param.MaxValue)
    setPassed(params, "max_value")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinValue != 0 {
    setParamInt(params, "min_value", param.MinValue)
    setPassed(params, "min_value")
  }

  // Detect if the parameter was passed; set if so.
  if param.ScalerMethod != "standard_scaler" {
    setParamString(params, "scaler_method", param.ScalerMethod)
    setPassed(params, "scaler_method")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt(params, "seed", param.Seed)
    setPassed(params, "seed")
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

  // Call the mlpack program.
  C.mlpackPreprocessScale(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat(params, "output")
  var outputModel scalingModel
  outputModel.getScalingModel(params, "output_model")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return output, outputModel
}
