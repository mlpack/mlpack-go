package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_nca
#include <capi/nca.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type NcaOptionalParam struct {
    ArmijoConstant float64
    BatchSize int
    Labels *mat.Dense
    LinearScan bool
    MaxIterations int
    MaxLineSearchTrials int
    MaxStep float64
    MinStep float64
    Normalize bool
    NumBasis int
    Optimizer string
    Seed int
    StepSize float64
    Tolerance float64
    Verbose bool
    Wolfe float64
}

func NcaOptions() *NcaOptionalParam {
  return &NcaOptionalParam{
    ArmijoConstant: 0.0001,
    BatchSize: 50,
    Labels: nil,
    LinearScan: false,
    MaxIterations: 500000,
    MaxLineSearchTrials: 50,
    MaxStep: 1e+20,
    MinStep: 1e-20,
    Normalize: false,
    NumBasis: 5,
    Optimizer: "sgd",
    Seed: 0,
    StepSize: 0.01,
    Tolerance: 1e-07,
    Verbose: false,
    Wolfe: 0.9,
  }
}

/*
  This program implements Neighborhood Components Analysis, both a linear
  dimensionality reduction technique and a distance learning technique.  The
  method seeks to improve k-nearest-neighbor classification on a dataset by
  scaling the dimensions.  The method is nonparametric, and does not require a
  value of k.  It works by using stochastic ("soft") neighbor assignments and
  using optimization techniques over the gradient of the accuracy of the
  neighbor assignments.
  
  To work, this algorithm needs labeled data.  It can be given as the last row
  of the input dataset (specified with "Input"), or alternatively as a separate
  matrix (specified with "Labels").
  
  This implementation of NCA uses stochastic gradient descent, mini-batch
  stochastic gradient descent, or the L_BFGS optimizer.  These optimizers do not
  guarantee global convergence for a nonconvex objective function (NCA's
  objective function is nonconvex), so the final results could depend on the
  random seed or other optimizer parameters.
  
  Stochastic gradient descent, specified by the value 'sgd' for the parameter
  "Optimizer", depends primarily on three parameters: the step size (specified
  with "StepSize"), the batch size (specified with "BatchSize"), and the maximum
  number of iterations (specified with "MaxIterations").  In addition, a
  normalized starting point can be used by specifying the "Normalize" parameter,
  which is necessary if many warnings of the form 'Denominator of p_i is 0!' are
  given.  Tuning the step size can be a tedious affair.  In general, the step
  size is too large if the objective is not mostly uniformly decreasing, or if
  zero-valued denominator warnings are being issued.  The step size is too small
  if the objective is changing very slowly.  Setting the termination condition
  can be done easily once a good step size parameter is found; either increase
  the maximum iterations to a large number and allow SGD to find a minimum, or
  set the maximum iterations to 0 (allowing infinite iterations) and set the
  tolerance (specified by "Tolerance") to define the maximum allowed difference
  between objectives for SGD to terminate.  Be careful---setting the tolerance
  instead of the maximum iterations can take a very long time and may actually
  never converge due to the properties of the SGD optimizer. Note that a single
  iteration of SGD refers to a single point, so to take a single pass over the
  dataset, set the value of the "MaxIterations" parameter equal to the number of
  points in the dataset.
  
  The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter
  "Optimizer", uses a back-tracking line search algorithm to minimize a
  function.  The following parameters are used by L-BFGS: "NumBasis" (specifies
  the number of memory points used by L-BFGS), "MaxIterations",
  "ArmijoConstant", "Wolfe", "Tolerance" (the optimization is terminated when
  the gradient norm is below this value), "MaxLineSearchTrials", "MinStep", and
  "MaxStep" (which both refer to the line search routine).  For more details on
  the L-BFGS optimizer, consult either the mlpack L-BFGS documentation (in
  lbfgs.hpp) or the vast set of published literature on L-BFGS.
  
  By default, the SGD optimizer is used.


  Input parameters:

   - input (mat.Dense): Input dataset to run NCA on.
   - ArmijoConstant (float64): Armijo constant for L-BFGS.  Default value
        0.0001.
   - BatchSize (int): Batch size for mini-batch SGD.  Default value 50.
   - Labels (mat.Dense): Labels for input dataset.
   - LinearScan (bool): Don't shuffle the order in which data points are
        visited for SGD or mini-batch SGD.
   - MaxIterations (int): Maximum number of iterations for SGD or L-BFGS
        (0 indicates no limit).  Default value 500000.
   - MaxLineSearchTrials (int): Maximum number of line search trials for
        L-BFGS.  Default value 50.
   - MaxStep (float64): Maximum step of line search for L-BFGS.  Default
        value 1e+20.
   - MinStep (float64): Minimum step of line search for L-BFGS.  Default
        value 1e-20.
   - Normalize (bool): Use a normalized starting point for optimization.
        This is useful for when points are far apart, or when SGD is returning
        NaN.
   - NumBasis (int): Number of memory points to be stored for L-BFGS. 
        Default value 5.
   - Optimizer (string): Optimizer to use; 'sgd' or 'lbfgs'.  Default
        value 'sgd'.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - StepSize (float64): Step size for stochastic gradient descent
        (alpha).  Default value 0.01.
   - Tolerance (float64): Maximum tolerance for termination of SGD or
        L-BFGS.  Default value 1e-07.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.
   - Wolfe (float64): Wolfe condition parameter for L-BFGS.  Default value
        0.9.

  Output parameters:

   - output (mat.Dense): Output matrix for learned distance matrix.

 */
func Nca(input *mat.Dense, param *NcaOptionalParam) (*mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Neighborhood Components Analysis (NCA)")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.ArmijoConstant != 0.0001 {
    setParamDouble("armijo_constant", param.ArmijoConstant)
    setPassed("armijo_constant")
  }

  // Detect if the parameter was passed; set if so.
  if param.BatchSize != 50 {
    setParamInt("batch_size", param.BatchSize)
    setPassed("batch_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow("labels", param.Labels)
    setPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.LinearScan != false {
    setParamBool("linear_scan", param.LinearScan)
    setPassed("linear_scan")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 500000 {
    setParamInt("max_iterations", param.MaxIterations)
    setPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxLineSearchTrials != 50 {
    setParamInt("max_line_search_trials", param.MaxLineSearchTrials)
    setPassed("max_line_search_trials")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxStep != 1e+20 {
    setParamDouble("max_step", param.MaxStep)
    setPassed("max_step")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinStep != 1e-20 {
    setParamDouble("min_step", param.MinStep)
    setPassed("min_step")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalize != false {
    setParamBool("normalize", param.Normalize)
    setPassed("normalize")
  }

  // Detect if the parameter was passed; set if so.
  if param.NumBasis != 5 {
    setParamInt("num_basis", param.NumBasis)
    setPassed("num_basis")
  }

  // Detect if the parameter was passed; set if so.
  if param.Optimizer != "sgd" {
    setParamString("optimizer", param.Optimizer)
    setPassed("optimizer")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.StepSize != 0.01 {
    setParamDouble("step_size", param.StepSize)
    setPassed("step_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tolerance != 1e-07 {
    setParamDouble("tolerance", param.Tolerance)
    setPassed("tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Detect if the parameter was passed; set if so.
  if param.Wolfe != 0.9 {
    setParamDouble("wolfe", param.Wolfe)
    setPassed("wolfe")
  }

  // Mark all output options as passed.
  setPassed("output")

  // Call the mlpack program.
  C.mlpackNca()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat("output")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return output
}
