package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_fastmks
#include <capi/fastmks.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

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

type fastmksModel struct {
  mem unsafe.Pointer
}

func (m *fastmksModel) allocFastMKSModel(identifier string) {
  m.mem = C.mlpackGetFastMKSModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *fastmksModel) getFastMKSModel(identifier string) {
  m.allocFastMKSModel(identifier)
}

func setFastMKSModel(identifier string, ptr *fastmksModel) {
  C.mlpackSetFastMKSModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
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
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("FastMKS (Fast Max-Kernel Search)")

  // Detect if the parameter was passed; set if so.
  if param.Bandwidth != 1 {
    setParamDouble("bandwidth", param.Bandwidth)
    setPassed("bandwidth")
  }

  // Detect if the parameter was passed; set if so.
  if param.Base != 2 {
    setParamDouble("base", param.Base)
    setPassed("base")
  }

  // Detect if the parameter was passed; set if so.
  if param.Degree != 2 {
    setParamDouble("degree", param.Degree)
    setPassed("degree")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setFastMKSModel("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 0 {
    setParamInt("k", param.K)
    setPassed("k")
  }

  // Detect if the parameter was passed; set if so.
  if param.Kernel != "linear" {
    setParamString("kernel", param.Kernel)
    setPassed("kernel")
  }

  // Detect if the parameter was passed; set if so.
  if param.Naive != false {
    setParamBool("naive", param.Naive)
    setPassed("naive")
  }

  // Detect if the parameter was passed; set if so.
  if param.Offset != 0 {
    setParamDouble("offset", param.Offset)
    setPassed("offset")
  }

  // Detect if the parameter was passed; set if so.
  if param.Query != nil {
    gonumToArmaMat("query", param.Query)
    setPassed("query")
  }

  // Detect if the parameter was passed; set if so.
  if param.Reference != nil {
    gonumToArmaMat("reference", param.Reference)
    setPassed("reference")
  }

  // Detect if the parameter was passed; set if so.
  if param.Scale != 1 {
    setParamDouble("scale", param.Scale)
    setPassed("scale")
  }

  // Detect if the parameter was passed; set if so.
  if param.Single != false {
    setParamBool("single", param.Single)
    setPassed("single")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("indices")
  setPassed("kernels")
  setPassed("output_model")

  // Call the mlpack program.
  C.mlpackFastmks()

  // Initialize result variable and get output.
  var indicesPtr mlpackArma
  indices := indicesPtr.armaToGonumUmat("indices")
  var kernelsPtr mlpackArma
  kernels := kernelsPtr.armaToGonumMat("kernels")
  var outputModel fastmksModel
  outputModel.getFastMKSModel("output_model")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return indices, kernels, outputModel
}
