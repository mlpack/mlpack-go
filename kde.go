package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_kde
#include <capi/kde.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type KdeOptionalParam struct {
    AbsError float64
    Algorithm string
    Bandwidth float64
    InitialSampleSize int
    InputModel *kdeModel
    Kernel string
    McBreakCoef float64
    McEntryCoef float64
    McProbability float64
    MonteCarlo bool
    Query *mat.Dense
    Reference *mat.Dense
    RelError float64
    Tree string
    Verbose bool
}

func KdeOptions() *KdeOptionalParam {
  return &KdeOptionalParam{
    AbsError: 0,
    Algorithm: "dual-tree",
    Bandwidth: 1,
    InitialSampleSize: 100,
    InputModel: nil,
    Kernel: "gaussian",
    McBreakCoef: 0.4,
    McEntryCoef: 3,
    McProbability: 0.95,
    MonteCarlo: false,
    Query: nil,
    Reference: nil,
    RelError: 0.05,
    Tree: "kd-tree",
    Verbose: false,
  }
}

/*
  This program performs a Kernel Density Estimation. KDE is a non-parametric way
  of estimating probability density function. For each query point the program
  will estimate its probability density by applying a kernel function to each
  reference point. The computational complexity of this is O(N^2) where there
  are N query points and N reference points, but this implementation will
  typically see better performance as it uses an approximate dual or single tree
  algorithm for acceleration.
  
  Dual or single tree optimization avoids many barely relevant calculations (as
  kernel function values decrease with distance), so it is an approximate
  computation. You can specify the maximum relative error tolerance for each
  query value with "RelError" as well as the maximum absolute error tolerance
  with the parameter "AbsError". This program runs using an Euclidean metric.
  Kernel function can be selected using the "Kernel" option. You can also choose
  what which type of tree to use for the dual-tree algorithm with "Tree". It is
  also possible to select whether to use dual-tree algorithm or single-tree
  algorithm using the "Algorithm" option.
  
  Monte Carlo estimations can be used to accelerate the KDE estimate when the
  Gaussian Kernel is used. This provides a probabilistic guarantee on the the
  error of the resulting KDE instead of an absolute guarantee.To enable Monte
  Carlo estimations, the "MonteCarlo" flag can be used, and success probability
  can be set with the "McProbability" option. It is possible to set the initial
  sample size for the Monte Carlo estimation using "InitialSampleSize". This
  implementation will only consider a node, as a candidate for the Monte Carlo
  estimation, if its number of descendant nodes is bigger than the initial
  sample size. This can be controlled using a coefficient that will multiply the
  initial sample size and can be set using "McEntryCoef". To avoid using the
  same amount of computations an exact approach would take, this program
  recurses the tree whenever a fraction of the amount of the node's descendant
  points have already been computed. This fraction is set using "McBreakCoef".

  For example, the following will run KDE using the data in ref_data for
  training and the data in qu_data as query data. It will apply an Epanechnikov
  kernel with a 0.2 bandwidth to each reference point and use a KD-Tree for the
  dual-tree optimization. The returned predictions will be within 5% of the real
  KDE value for each query point.
  
  // Initialize optional parameters for Kde().
  param := mlpack.KdeOptions()
  param.Reference = ref_data
  param.Query = qu_data
  param.Bandwidth = 0.2
  param.Kernel = "epanechnikov"
  param.Tree = "kd-tree"
  param.RelError = 0.05
  
  _, out_data := mlpack.Kde(param)
  
  the predicted density estimations will be stored in out_data.
  If no "Query" is provided, then KDE will be computed on the "Reference"
  dataset.
  It is possible to select either a reference dataset or an input model but not
  both at the same time. If an input model is selected and parameter values are
  not set (e.g. "Bandwidth") then default parameter values will be used.
  
  In addition to the last program call, it is also possible to activate Monte
  Carlo estimations if a Gaussian kernel is used. This can provide faster
  results, but the KDE will only have a probabilistic guarantee of meeting the
  desired error bound (instead of an absolute guarantee). The following example
  will run KDE using a Monte Carlo estimation when possible. The results will be
  within a 5% of the real KDE value with a 95% probability. Initial sample size
  for the Monte Carlo estimation will be 200 points and a node will be a
  candidate for the estimation only when it contains 700 (i.e. 3.5*200) points.
  If a node contains 700 points and 420 (i.e. 0.6*700) have already been
  sampled, then the algorithm will recurse instead of keep sampling.
  
  // Initialize optional parameters for Kde().
  param := mlpack.KdeOptions()
  param.Reference = ref_data
  param.Query = qu_data
  param.Bandwidth = 0.2
  param.Kernel = "gaussian"
  param.Tree = "kd-tree"
  param.RelError = 0.05
  param.MonteCarlo = 
  param.McProbability = 0.95
  param.InitialSampleSize = 200
  param.McEntryCoef = 3.5
  param.McBreakCoef = 0.6
  
  _, out_data := mlpack.Kde(param)

  Input parameters:

   - AbsError (float64): Relative error tolerance for the prediction. 
        Default value 0.
   - Algorithm (string): Algorithm to use for the prediction.('dual-tree',
        'single-tree').  Default value 'dual-tree'.
   - Bandwidth (float64): Bandwidth of the kernel.  Default value 1.
   - InitialSampleSize (int): Initial sample size for Monte Carlo
        estimations.  Default value 100.
   - InputModel (kdeModel): Contains pre-trained KDE model.
   - Kernel (string): Kernel to use for the prediction.('gaussian',
        'epanechnikov', 'laplacian', 'spherical', 'triangular').  Default value
        'gaussian'.
   - McBreakCoef (float64): Controls what fraction of the amount of node's
        descendants is the limit for the sample size before it recurses. 
        Default value 0.4.
   - McEntryCoef (float64): Controls how much larger does the amount of
        node descendants has to be compared to the initial sample size in order
        to be a candidate for Monte Carlo estimations.  Default value 3.
   - McProbability (float64): Probability of the estimation being bounded
        by relative error when using Monte Carlo estimations.  Default value
        0.95.
   - MonteCarlo (bool): Whether to use Monte Carlo estimations when
        possible.
   - Query (mat.Dense): Query dataset to KDE on.
   - Reference (mat.Dense): Input reference dataset use for KDE.
   - RelError (float64): Relative error tolerance for the prediction. 
        Default value 0.05.
   - Tree (string): Tree to use for the prediction.('kd-tree',
        'ball-tree', 'cover-tree', 'octree', 'r-tree').  Default value
        'kd-tree'.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (kdeModel): If specified, the KDE model will be saved
        here.
   - predictions (mat.Dense): Vector to store density predictions.

 */
func Kde(param *KdeOptionalParam) (kdeModel, *mat.Dense) {
  params := getParams("kde")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.AbsError != 0 {
    setParamDouble(params, "abs_error", param.AbsError)
    setPassed(params, "abs_error")
  }

  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "dual-tree" {
    setParamString(params, "algorithm", param.Algorithm)
    setPassed(params, "algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.Bandwidth != 1 {
    setParamDouble(params, "bandwidth", param.Bandwidth)
    setPassed(params, "bandwidth")
  }

  // Detect if the parameter was passed; set if so.
  if param.InitialSampleSize != 100 {
    setParamInt(params, "initial_sample_size", param.InitialSampleSize)
    setPassed(params, "initial_sample_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setKDEModel(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Kernel != "gaussian" {
    setParamString(params, "kernel", param.Kernel)
    setPassed(params, "kernel")
  }

  // Detect if the parameter was passed; set if so.
  if param.McBreakCoef != 0.4 {
    setParamDouble(params, "mc_break_coef", param.McBreakCoef)
    setPassed(params, "mc_break_coef")
  }

  // Detect if the parameter was passed; set if so.
  if param.McEntryCoef != 3 {
    setParamDouble(params, "mc_entry_coef", param.McEntryCoef)
    setPassed(params, "mc_entry_coef")
  }

  // Detect if the parameter was passed; set if so.
  if param.McProbability != 0.95 {
    setParamDouble(params, "mc_probability", param.McProbability)
    setPassed(params, "mc_probability")
  }

  // Detect if the parameter was passed; set if so.
  if param.MonteCarlo != false {
    setParamBool(params, "monte_carlo", param.MonteCarlo)
    setPassed(params, "monte_carlo")
  }

  // Detect if the parameter was passed; set if so.
  if param.Query != nil {
    gonumToArmaMat(params, "query", param.Query, false)
    setPassed(params, "query")
  }

  // Detect if the parameter was passed; set if so.
  if param.Reference != nil {
    gonumToArmaMat(params, "reference", param.Reference, false)
    setPassed(params, "reference")
  }

  // Detect if the parameter was passed; set if so.
  if param.RelError != 0.05 {
    setParamDouble(params, "rel_error", param.RelError)
    setPassed(params, "rel_error")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tree != "kd-tree" {
    setParamString(params, "tree", param.Tree)
    setPassed(params, "tree")
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

  // Call the mlpack program.
  C.mlpackKde(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputModel kdeModel
  outputModel.getKDEModel(params, "output_model")
  var predictionsPtr mlpackArma
  predictions := predictionsPtr.armaToGonumCol(params, "predictions")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return outputModel, predictions
}
