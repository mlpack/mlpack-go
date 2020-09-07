package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_gmm_train
#include <capi/gmm_train.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type GmmTrainOptionalParam struct {
    DiagonalCovariance bool
    InputModel *gmm
    KmeansMaxIterations int
    MaxIterations int
    NoForcePositive bool
    Noise float64
    Percentage float64
    RefinedStart bool
    Samplings int
    Seed int
    Tolerance float64
    Trials int
    Verbose bool
}

func GmmTrainOptions() *GmmTrainOptionalParam {
  return &GmmTrainOptionalParam{
    DiagonalCovariance: false,
    InputModel: nil,
    KmeansMaxIterations: 1000,
    MaxIterations: 250,
    NoForcePositive: false,
    Noise: 0,
    Percentage: 0.02,
    RefinedStart: false,
    Samplings: 100,
    Seed: 0,
    Tolerance: 1e-10,
    Trials: 1,
    Verbose: false,
  }
}

/*
  This program takes a parametric estimate of a Gaussian mixture model (GMM)
  using the EM algorithm to find the maximum likelihood estimate.  The model may
  be saved and reused by other mlpack GMM tools.
  
  The input data to train on must be specified with the "Input" parameter, and
  the number of Gaussians in the model must be specified with the "Gaussians"
  parameter.  Optionally, many trials with different random initializations may
  be run, and the result with highest log-likelihood on the training data will
  be taken.  The number of trials to run is specified with the "Trials"
  parameter.  By default, only one trial is run.
  
  The tolerance for convergence and maximum number of iterations of the EM
  algorithm are specified with the "Tolerance" and "MaxIterations" parameters,
  respectively.  The GMM may be initialized for training with another model,
  specified with the "InputModel" parameter. Otherwise, the model is initialized
  by running k-means on the data.  The k-means clustering initialization can be
  controlled with the "KmeansMaxIterations", "RefinedStart", "Samplings", and
  "Percentage" parameters.  If "RefinedStart" is specified, then the
  Bradley-Fayyad refined start initialization will be used.  This can often lead
  to better clustering results.
  
  The 'diagonal_covariance' flag will cause the learned covariances to be
  diagonal matrices.  This significantly simplifies the model itself and causes
  training to be faster, but restricts the ability to fit more complex GMMs.
  
  If GMM training fails with an error indicating that a covariance matrix could
  not be inverted, make sure that the "NoForcePositive" parameter is not
  specified.  Alternately, adding a small amount of Gaussian noise (using the
  "Noise" parameter) to the entire dataset may help prevent Gaussians with zero
  variance in a particular dimension, which is usually the cause of
  non-invertible covariance matrices.
  
  The "NoForcePositive" parameter, if set, will avoid the checks after each
  iteration of the EM algorithm which ensure that the covariance matrices are
  positive definite.  Specifying the flag can cause faster runtime, but may also
  cause non-positive definite covariance matrices, which will cause the program
  to crash.

  As an example, to train a 6-Gaussian GMM on the data in data with a maximum of
  100 iterations of EM and 3 trials, saving the trained GMM to gmm, the
  following command can be used:
  
  // Initialize optional parameters for GmmTrain().
  param := mlpack.GmmTrainOptions()
  param.Trials = 3
  
  gmm := mlpack.GmmTrain(data, 6, param)
  
  To re-train that GMM on another set of data data2, the following command may
  be used: 
  
  // Initialize optional parameters for GmmTrain().
  param := mlpack.GmmTrainOptions()
  param.InputModel = &gmm
  
  new_gmm := mlpack.GmmTrain(data2, 6, param)

  Input parameters:

   - gaussians (int): Number of Gaussians in the GMM.
   - input (mat.Dense): The training data on which the model will be fit.
   - DiagonalCovariance (bool): Force the covariance of the Gaussians to
        be diagonal.  This can accelerate training time significantly.
   - InputModel (gmm): Initial input GMM model to start training with.
   - KmeansMaxIterations (int): Maximum number of iterations for the
        k-means algorithm (used to initialize EM).  Default value 1000.
   - MaxIterations (int): Maximum number of iterations of EM algorithm
        (passing 0 will run until convergence).  Default value 250.
   - NoForcePositive (bool): Do not force the covariance matrices to be
        positive definite.
   - Noise (float64): Variance of zero-mean Gaussian noise to add to data.
         Default value 0.
   - Percentage (float64): If using --refined_start, specify the
        percentage of the dataset used for each sampling (should be between 0.0
        and 1.0).  Default value 0.02.
   - RefinedStart (bool): During the initialization, use refined initial
        positions for k-means clustering (Bradley and Fayyad, 1998).
   - Samplings (int): If using --refined_start, specify the number of
        samplings used for initial points.  Default value 100.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - Tolerance (float64): Tolerance for convergence of EM.  Default value
        1e-10.
   - Trials (int): Number of trials to perform in training GMM.  Default
        value 1.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (gmm): Output for trained GMM model.

 */
func GmmTrain(gaussians int, input *mat.Dense, param *GmmTrainOptionalParam) (gmm) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Gaussian Mixture Model (GMM) Training")

  // Detect if the parameter was passed; set if so.
  setParamInt("gaussians", gaussians)
  setPassed("gaussians")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.DiagonalCovariance != false {
    setParamBool("diagonal_covariance", param.DiagonalCovariance)
    setPassed("diagonal_covariance")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setGMM("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.KmeansMaxIterations != 1000 {
    setParamInt("kmeans_max_iterations", param.KmeansMaxIterations)
    setPassed("kmeans_max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 250 {
    setParamInt("max_iterations", param.MaxIterations)
    setPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.NoForcePositive != false {
    setParamBool("no_force_positive", param.NoForcePositive)
    setPassed("no_force_positive")
  }

  // Detect if the parameter was passed; set if so.
  if param.Noise != 0 {
    setParamDouble("noise", param.Noise)
    setPassed("noise")
  }

  // Detect if the parameter was passed; set if so.
  if param.Percentage != 0.02 {
    setParamDouble("percentage", param.Percentage)
    setPassed("percentage")
  }

  // Detect if the parameter was passed; set if so.
  if param.RefinedStart != false {
    setParamBool("refined_start", param.RefinedStart)
    setPassed("refined_start")
  }

  // Detect if the parameter was passed; set if so.
  if param.Samplings != 100 {
    setParamInt("samplings", param.Samplings)
    setPassed("samplings")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tolerance != 1e-10 {
    setParamDouble("tolerance", param.Tolerance)
    setPassed("tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Trials != 1 {
    setParamInt("trials", param.Trials)
    setPassed("trials")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("output_model")

  // Call the mlpack program.
  C.mlpackGmmTrain()

  // Initialize result variable and get output.
  var outputModel gmm
  outputModel.getGMM("output_model")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return outputModel
}
