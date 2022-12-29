package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_fastmks
#include <capi/fastmks.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type FastmksOptionalParam struct {
    Bandwidth float64
    Base float64
    Degree float64
    InputModel *fastmksModel
    K int
    Kernel string
    Naive bool
    Offset float64
    Query *mat.Dense
    Reference *mat.Dense
    Scale float64
    Single bool
    Verbose bool
}

func FastmksOptions() *FastmksOptionalParam {
  return &FastmksOptionalParam{
    Bandwidth: 1,
    Base: 2,
    Degree: 2,
    InputModel: nil,
    K: 0,
    Kernel: "linear",
    Naive: false,
    Offset: 0,
    Query: nil,
    Reference: nil,
    Scale: 1,
    Single: false,
    Verbose: false,
  }
}

/*
  This program will find the k maximum kernels of a set of points, using a query
  set and a reference set (which can optionally be the same set). More
  specifically, for each point in the query set, the k points in the reference
  set with maximum kernel evaluations are found.  The kernel function used is
  specified with the "Kernel" parameter.

  For example, the following command will calculate, for each point in the query
  set query, the five points in the reference set reference with maximum kernel
  evaluation using the linear kernel.  The kernel evaluations may be saved with
  the  kernels output parameter and the indices may be saved with the indices
  output parameter.
  
  // Initialize optional parameters for Fastmks().
  param := mlpack.FastmksOptions()
  param.K = 5
  param.Reference = reference
  param.Query = query
  param.Kernel = "linear"
  
  indices, kernels, _ := mlpack.Fastmks(param)
  
  The output matrices are organized such that row i and column j in the indices
  matrix corresponds to the index of the point in the reference set that has
  j'th largest kernel evaluation with the point in the query set with index i. 
  Row i and column j in the kernels matrix corresponds to the kernel evaluation
  between those two points.
  
  This program performs FastMKS using a cover tree.  The base used to build the
  cover tree can be specified with the "Base" parameter.

  Input parameters:

   - Bandwidth (float64): Bandwidth (for Gaussian, Epanechnikov, and
        triangular kernels).  Default value 1.
   - Base (float64): Base to use during cover tree construction.  Default
        value 2.
   - Degree (float64): Degree of polynomial kernel.  Default value 2.
   - InputModel (fastmksModel): Input FastMKS model to use.
   - K (int): Number of maximum kernels to find.  Default value 0.
   - Kernel (string): Kernel type to use: 'linear', 'polynomial',
        'cosine', 'gaussian', 'epanechnikov', 'triangular', 'hyptan'.  Default
        value 'linear'.
   - Naive (bool): If true, O(n^2) naive mode is used for computation.
   - Offset (float64): Offset of kernel (for polynomial and hyptan
        kernels).  Default value 0.
   - Query (mat.Dense): The query dataset.
   - Reference (mat.Dense): The reference dataset.
   - Scale (float64): Scale of kernel (for hyptan kernel).  Default value
        1.
   - Single (bool): If true, single-tree search is used (as opposed to
        dual-tree search.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - indices (mat.Dense): Output matrix of indices.
   - kernels (mat.Dense): Output matrix of kernels.
   - outputModel (fastmksModel): Output for FastMKS model.

 */
func Fastmks(param *FastmksOptionalParam) (*mat.Dense, *mat.Dense, fastmksModel) {
  params := getParams("fastmks")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.Bandwidth != 1 {
    setParamDouble(params, "bandwidth", param.Bandwidth)
    setPassed(params, "bandwidth")
  }

  // Detect if the parameter was passed; set if so.
  if param.Base != 2 {
    setParamDouble(params, "base", param.Base)
    setPassed(params, "base")
  }

  // Detect if the parameter was passed; set if so.
  if param.Degree != 2 {
    setParamDouble(params, "degree", param.Degree)
    setPassed(params, "degree")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setFastMKSModel(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 0 {
    setParamInt(params, "k", param.K)
    setPassed(params, "k")
  }

  // Detect if the parameter was passed; set if so.
  if param.Kernel != "linear" {
    setParamString(params, "kernel", param.Kernel)
    setPassed(params, "kernel")
  }

  // Detect if the parameter was passed; set if so.
  if param.Naive != false {
    setParamBool(params, "naive", param.Naive)
    setPassed(params, "naive")
  }

  // Detect if the parameter was passed; set if so.
  if param.Offset != 0 {
    setParamDouble(params, "offset", param.Offset)
    setPassed(params, "offset")
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
  if param.Scale != 1 {
    setParamDouble(params, "scale", param.Scale)
    setPassed(params, "scale")
  }

  // Detect if the parameter was passed; set if so.
  if param.Single != false {
    setParamBool(params, "single", param.Single)
    setPassed(params, "single")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "indices")
  setPassed(params, "kernels")
  setPassed(params, "output_model")

  // Call the mlpack program.
  C.mlpackFastmks(params.mem, timers.mem)

  // Initialize result variable and get output.
  var indicesPtr mlpackArma
  indices := indicesPtr.armaToGonumUmat(params, "indices")
  var kernelsPtr mlpackArma
  kernels := kernelsPtr.armaToGonumMat(params, "kernels")
  var outputModel fastmksModel
  outputModel.getFastMKSModel(params, "output_model")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return indices, kernels, outputModel
}
