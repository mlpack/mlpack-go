package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_lmnn
#include <capi/lmnn.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type LmnnOptionalParam struct {
    BatchSize int
    Center bool
    Distance *mat.Dense
    K int
    Labels *mat.Dense
    LinearScan bool
    MaxIterations int
    Normalize bool
    Optimizer string
    Passes int
    PrintAccuracy bool
    Range int
    Rank int
    Regularization float64
    Seed int
    StepSize float64
    Tolerance float64
    Verbose bool
}

func LmnnOptions() *LmnnOptionalParam {
  return &LmnnOptionalParam{
    BatchSize: 50,
    Center: false,
    Distance: nil,
    K: 1,
    Labels: nil,
    LinearScan: false,
    MaxIterations: 100000,
    Normalize: false,
    Optimizer: "amsgrad",
    Passes: 50,
    PrintAccuracy: false,
    Range: 1,
    Rank: 0,
    Regularization: 0.5,
    Seed: 0,
    StepSize: 0.01,
    Tolerance: 1e-07,
    Verbose: false,
  }
}

/*
  This program implements Large Margin Nearest Neighbors, a distance learning
  technique.  The method seeks to improve k-nearest-neighbor classification on a
  dataset.  The method employes the strategy of reducing distance between
  similar labeled data points (a.k.a target neighbors) and increasing distance
  between differently labeled points (a.k.a impostors) using standard
  optimization techniques over the gradient of the distance between data points.
  
  To work, this algorithm needs labeled data.  It can be given as the last row
  of the input dataset (specified with "Input"), or alternatively as a separate
  matrix (specified with "Labels").  Additionally, a starting point for
  optimization (specified with "Distance"can be given, having (r x d)
  dimensionality.  Here r should satisfy 1 <= r <= d, Consequently a Low-Rank
  matrix will be optimized. Alternatively, Low-Rank distance can be learned by
  specifying the "Rank"parameter (A Low-Rank matrix with uniformly distributed
  values will be used as initial learning point). 
  
  The program also requires number of targets neighbors to work with ( specified
  with "K"), A regularization parameter can also be passed, It acts as a trade
  of between the pulling and pushing terms (specified with "Regularization"), In
  addition, this implementation of LMNN includes a parameter to decide the
  interval after which impostors must be re-calculated (specified with "Range").
  
  Output can either be the learned distance matrix (specified with "Output"), or
  the transformed dataset  (specified with "TransformedData"), or both.
  Additionally mean-centered dataset (specified with "CenteredData") can be
  accessed given mean-centering (specified with "Center") is performed on the
  dataset. Accuracy on initial dataset and final transformed dataset can be
  printed by specifying the "PrintAccuracy"parameter. 
  
  This implementation of LMNN uses AdaGrad, BigBatch_SGD, stochastic gradient
  descent, mini-batch stochastic gradient descent, or the L_BFGS optimizer. 
  
  AdaGrad, specified by the value 'adagrad' for the parameter "Optimizer", uses
  maximum of past squared gradients. It primarily on six parameters: the step
  size (specified with "StepSize"), the batch size (specified with "BatchSize"),
  the maximum number of passes (specified with "Passes"). Inaddition, a
  normalized starting point can be used by specifying the "Normalize" parameter.
  
  
  BigBatch_SGD, specified by the value 'bbsgd' for the parameter "Optimizer",
  depends primarily on four parameters: the step size (specified with
  "StepSize"), the batch size (specified with "BatchSize"), the maximum number
  of passes (specified with "Passes").  In addition, a normalized starting point
  can be used by specifying the "Normalize" parameter. 
  
  Stochastic gradient descent, specified by the value 'sgd' for the parameter
  "Optimizer", depends primarily on three parameters: the step size (specified
  with "StepSize"), the batch size (specified with "BatchSize"), and the maximum
  number of passes (specified with "Passes").  In addition, a normalized
  starting point can be used by specifying the "Normalize" parameter.
  Furthermore, mean-centering can be performed on the dataset by specifying the
  "Center"parameter. 
  
  The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter
  "Optimizer", uses a back-tracking line search algorithm to minimize a
  function.  The following parameters are used by L-BFGS: "MaxIterations",
  "Tolerance"(the optimization is terminated when the gradient norm is below
  this value).  For more details on the L-BFGS optimizer, consult either the
  mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of published
  literature on L-BFGS.  In addition, a normalized starting point can be used by
  specifying the "Normalize" parameter.
  
  By default, the AMSGrad optimizer is used.

  Example - Let's say we want to learn distance on iris dataset with number of
  targets as 3 using BigBatch_SGD optimizer. A simple call for the same will
  look like: 
  
  // Initialize optional parameters for MlpackLmnn().
  param := mlpack.MlpackLmnnOptions()
  param.Labels = iris_labels
  param.K = 3
  param.Optimizer = "bbsgd"
  
  _, output, _ := mlpack.MlpackLmnn(iris, param)
  
  An another program call making use of range & regularization parameter with
  dataset having labels as last column can be made as: 
  
  // Initialize optional parameters for MlpackLmnn().
  param := mlpack.MlpackLmnnOptions()
  param.K = 5
  param.Range = 10
  param.Regularization = 0.4
  
  _, output, _ := mlpack.MlpackLmnn(letter_recognition, param)

  Input parameters:

   - input (mat.Dense): Input dataset to run LMNN on.
   - BatchSize (int): Batch size for mini-batch SGD.  Default value 50.
   - Center (bool): Perform mean-centering on the dataset. It is useful
        when the centroid of the data is far from the origin.
   - Distance (mat.Dense): Initial distance matrix to be used as starting
        point
   - K (int): Number of target neighbors to use for each datapoint. 
        Default value 1.
   - Labels (mat.Dense): Labels for input dataset.
   - LinearScan (bool): Don't shuffle the order in which data points are
        visited for SGD or mini-batch SGD.
   - MaxIterations (int): Maximum number of iterations for L-BFGS (0
        indicates no limit).  Default value 100000.
   - Normalize (bool): Use a normalized starting point for optimization.
        Itis useful for when points are far apart, or when SGD is returning
        NaN.
   - Optimizer (string): Optimizer to use; 'amsgrad', 'bbsgd', 'sgd', or
        'lbfgs'.  Default value 'amsgrad'.
   - Passes (int): Maximum number of full passes over dataset for AMSGrad,
        BB_SGD and SGD.  Default value 50.
   - PrintAccuracy (bool): Print accuracies on initial and transformed
        dataset
   - Range (int): Number of iterations after which impostors needs to be
        recalculated  Default value 1.
   - Rank (int): Rank of distance matrix to be optimized.   Default value
        0.
   - Regularization (float64): Regularization for LMNN objective function 
         Default value 0.5.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - StepSize (float64): Step size for AMSGrad, BB_SGD and SGD (alpha). 
        Default value 0.01.
   - Tolerance (float64): Maximum tolerance for termination of AMSGrad,
        BB_SGD, SGD or L-BFGS.  Default value 1e-07.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - centeredData (mat.Dense): Output matrix for mean-centered dataset.
   - output (mat.Dense): Output matrix for learned distance matrix.
   - transformedData (mat.Dense): Output matrix for transformed dataset.

 */
func Lmnn(input *mat.Dense, param *LmnnOptionalParam) (*mat.Dense, *mat.Dense, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Large Margin Nearest Neighbors (LMNN)")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.BatchSize != 50 {
    setParamInt("batch_size", param.BatchSize)
    setPassed("batch_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Center != false {
    setParamBool("center", param.Center)
    setPassed("center")
  }

  // Detect if the parameter was passed; set if so.
  if param.Distance != nil {
    gonumToArmaMat("distance", param.Distance)
    setPassed("distance")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 1 {
    setParamInt("k", param.K)
    setPassed("k")
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
  if param.MaxIterations != 100000 {
    setParamInt("max_iterations", param.MaxIterations)
    setPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalize != false {
    setParamBool("normalize", param.Normalize)
    setPassed("normalize")
  }

  // Detect if the parameter was passed; set if so.
  if param.Optimizer != "amsgrad" {
    setParamString("optimizer", param.Optimizer)
    setPassed("optimizer")
  }

  // Detect if the parameter was passed; set if so.
  if param.Passes != 50 {
    setParamInt("passes", param.Passes)
    setPassed("passes")
  }

  // Detect if the parameter was passed; set if so.
  if param.PrintAccuracy != false {
    setParamBool("print_accuracy", param.PrintAccuracy)
    setPassed("print_accuracy")
  }

  // Detect if the parameter was passed; set if so.
  if param.Range != 1 {
    setParamInt("range", param.Range)
    setPassed("range")
  }

  // Detect if the parameter was passed; set if so.
  if param.Rank != 0 {
    setParamInt("rank", param.Rank)
    setPassed("rank")
  }

  // Detect if the parameter was passed; set if so.
  if param.Regularization != 0.5 {
    setParamDouble("regularization", param.Regularization)
    setPassed("regularization")
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

  // Mark all output options as passed.
  setPassed("centered_data")
  setPassed("output")
  setPassed("transformed_data")

  // Call the mlpack program.
  C.mlpackLmnn()

  // Initialize result variable and get output.
  var centeredDataPtr mlpackArma
  centeredData := centeredDataPtr.armaToGonumMat("centered_data")
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat("output")
  var transformedDataPtr mlpackArma
  transformedData := transformedDataPtr.armaToGonumMat("transformed_data")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return centeredData, output, transformedData
}
