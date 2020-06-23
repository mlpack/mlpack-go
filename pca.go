package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_pca
#include <capi/pca.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type PcaOptionalParam struct {
    DecompositionMethod string
    NewDimensionality int
    Scale bool
    VarToRetain float64
    Verbose bool
}

func PcaOptions() *PcaOptionalParam {
  return &PcaOptionalParam{
    DecompositionMethod: "exact",
    NewDimensionality: 0,
    Scale: false,
    VarToRetain: 0,
    Verbose: false,
  }
}

/*
  This program performs principal components analysis on the given dataset using
  the exact, randomized, randomized block Krylov, or QUIC SVD method. It will
  transform the data onto its principal components, optionally performing
  dimensionality reduction by ignoring the principal components with the
  smallest eigenvalues.
  
  Use the "Input" parameter to specify the dataset to perform PCA on.  A desired
  new dimensionality can be specified with the "NewDimensionality" parameter, or
  the desired variance to retain can be specified with the "VarToRetain"
  parameter.  If desired, the dataset can be scaled before running PCA with the
  "Scale" parameter.
  
  Multiple different decomposition techniques can be used.  The method to use
  can be specified with the "DecompositionMethod" parameter, and it may take the
  values 'exact', 'randomized', or 'quic'.
  
  For example, to reduce the dimensionality of the matrix data to 5 dimensions
  using randomized SVD for the decomposition, storing the output matrix to
  data_mod, the following command can be used:
  
  // Initialize optional parameters for Pca().
  param := mlpack.PcaOptions()
  param.NewDimensionality = 5
  param.DecompositionMethod = "randomized"
  
  data_mod := mlpack.Pca(data, param)


  Input parameters:

   - input (mat.Dense): Input dataset to perform PCA on.
   - DecompositionMethod (string): Method used for the principal
        components analysis: 'exact', 'randomized', 'randomized-block-krylov',
        'quic'.  Default value 'exact'.
   - NewDimensionality (int): Desired dimensionality of output dataset. If
        0, no dimensionality reduction is performed.  Default value 0.
   - Scale (bool): If set, the data will be scaled before running PCA,
        such that the variance of each feature is 1.
   - VarToRetain (float64): Amount of variance to retain; should be
        between 0 and 1.  If 1, all variance is retained.  Overrides -d. 
        Default value 0.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix to save modified dataset to.

 */
func Pca(input *mat.Dense, param *PcaOptionalParam) (*mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Principal Components Analysis")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.DecompositionMethod != "exact" {
    setParamString("decomposition_method", param.DecompositionMethod)
    setPassed("decomposition_method")
  }

  // Detect if the parameter was passed; set if so.
  if param.NewDimensionality != 0 {
    setParamInt("new_dimensionality", param.NewDimensionality)
    setPassed("new_dimensionality")
  }

  // Detect if the parameter was passed; set if so.
  if param.Scale != false {
    setParamBool("scale", param.Scale)
    setPassed("scale")
  }

  // Detect if the parameter was passed; set if so.
  if param.VarToRetain != 0 {
    setParamDouble("var_to_retain", param.VarToRetain)
    setPassed("var_to_retain")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("output")

  // Call the mlpack program.
  C.mlpackPca()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat("output")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return output
}
