package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_kernel_pca
#include <capi/kernel_pca.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type KernelPcaOptionalParam struct {
    Bandwidth float64
    Center bool
    Degree float64
    KernelScale float64
    NewDimensionality int
    NystroemMethod bool
    Offset float64
    Sampling string
    Verbose bool
}

func KernelPcaOptions() *KernelPcaOptionalParam {
  return &KernelPcaOptionalParam{
    Bandwidth: 1,
    Center: false,
    Degree: 1,
    KernelScale: 1,
    NewDimensionality: 0,
    NystroemMethod: false,
    Offset: 0,
    Sampling: "kmeans",
    Verbose: false,
  }
}

/*
  This program performs Kernel Principal Components Analysis (KPCA) on the
  specified dataset with the specified kernel.  This will transform the data
  onto the kernel principal components, and optionally reduce the dimensionality
  by ignoring the kernel principal components with the smallest eigenvalues.
  
  For the case where a linear kernel is used, this reduces to regular PCA.
  
  The kernels that are supported are listed below:
  
   * 'linear': the standard linear dot product (same as normal PCA):
      K(x, y) = x^T y
  
   * 'gaussian': a Gaussian kernel; requires bandwidth:
      K(x, y) = exp(-(|| x - y || ^ 2) / (2 * (bandwidth ^ 2)))
  
   * 'polynomial': polynomial kernel; requires offset and degree:
      K(x, y) = (x^T y + offset) ^ degree
  
   * 'hyptan': hyperbolic tangent kernel; requires scale and offset:
      K(x, y) = tanh(scale * (x^T y) + offset)
  
   * 'laplacian': Laplacian kernel; requires bandwidth:
      K(x, y) = exp(-(|| x - y ||) / bandwidth)
  
   * 'epanechnikov': Epanechnikov kernel; requires bandwidth:
      K(x, y) = max(0, 1 - || x - y ||^2 / bandwidth^2)
  
   * 'cosine': cosine distance:
      K(x, y) = 1 - (x^T y) / (|| x || * || y ||)
  
  The parameters for each of the kernels should be specified with the options
  "Bandwidth", "KernelScale", "Offset", or "Degree" (or a combination of those
  parameters).
  
  Optionally, the Nystroem method ("Using the Nystroem method to speed up kernel
  machines", 2001) can be used to calculate the kernel matrix by specifying the
  "NystroemMethod" parameter. This approach works by using a subset of the data
  as basis to reconstruct the kernel matrix; to specify the sampling scheme, the
  "Sampling" parameter is used.  The sampling scheme for the Nystroem method can
  be chosen from the following list: 'kmeans', 'random', 'ordered'.

  For example, the following command will perform KPCA on the dataset input
  using the Gaussian kernel, and saving the transformed data to transformed: 
  
  // Initialize optional parameters for KernelPca().
  param := mlpack.KernelPcaOptions()
  
  transformed := mlpack.KernelPca(input, "gaussian", param)

  Input parameters:

   - input (mat.Dense): Input dataset to perform KPCA on.
   - kernel (string): The kernel to use; see the above documentation for
        the list of usable kernels.
   - Bandwidth (float64): Bandwidth, for 'gaussian' and 'laplacian'
        kernels.  Default value 1.
   - Center (bool): If set, the transformed data will be centered about
        the origin.
   - Degree (float64): Degree of polynomial, for 'polynomial' kernel. 
        Default value 1.
   - KernelScale (float64): Scale, for 'hyptan' kernel.  Default value 1.
   - NewDimensionality (int): If not 0, reduce the dimensionality of the
        output dataset by ignoring the dimensions with the smallest eigenvalues.
         Default value 0.
   - NystroemMethod (bool): If set, the Nystroem method will be used.
   - Offset (float64): Offset, for 'hyptan' and 'polynomial' kernels. 
        Default value 0.
   - Sampling (string): Sampling scheme to use for the Nystroem method:
        'kmeans', 'random', 'ordered'  Default value 'kmeans'.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix to save modified dataset to.

 */
func KernelPca(input *mat.Dense, kernel string, param *KernelPcaOptionalParam) (*mat.Dense) {
  params := getParams("kernel_pca")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  gonumToArmaMat(params, "input", input)
  setPassed(params, "input")

  // Detect if the parameter was passed; set if so.
  setParamString(params, "kernel", kernel)
  setPassed(params, "kernel")

  // Detect if the parameter was passed; set if so.
  if param.Bandwidth != 1 {
    setParamDouble(params, "bandwidth", param.Bandwidth)
    setPassed(params, "bandwidth")
  }

  // Detect if the parameter was passed; set if so.
  if param.Center != false {
    setParamBool(params, "center", param.Center)
    setPassed(params, "center")
  }

  // Detect if the parameter was passed; set if so.
  if param.Degree != 1 {
    setParamDouble(params, "degree", param.Degree)
    setPassed(params, "degree")
  }

  // Detect if the parameter was passed; set if so.
  if param.KernelScale != 1 {
    setParamDouble(params, "kernel_scale", param.KernelScale)
    setPassed(params, "kernel_scale")
  }

  // Detect if the parameter was passed; set if so.
  if param.NewDimensionality != 0 {
    setParamInt(params, "new_dimensionality", param.NewDimensionality)
    setPassed(params, "new_dimensionality")
  }

  // Detect if the parameter was passed; set if so.
  if param.NystroemMethod != false {
    setParamBool(params, "nystroem_method", param.NystroemMethod)
    setPassed(params, "nystroem_method")
  }

  // Detect if the parameter was passed; set if so.
  if param.Offset != 0 {
    setParamDouble(params, "offset", param.Offset)
    setPassed(params, "offset")
  }

  // Detect if the parameter was passed; set if so.
  if param.Sampling != "kmeans" {
    setParamString(params, "sampling", param.Sampling)
    setPassed(params, "sampling")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "output")

  // Call the mlpack program.
  C.mlpackKernelPca(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat(params, "output")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return output
}
