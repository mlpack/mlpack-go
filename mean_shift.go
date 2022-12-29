package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_mean_shift
#include <capi/mean_shift.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type MeanShiftOptionalParam struct {
    ForceConvergence bool
    InPlace bool
    LabelsOnly bool
    MaxIterations int
    Radius float64
    Verbose bool
}

func MeanShiftOptions() *MeanShiftOptionalParam {
  return &MeanShiftOptionalParam{
    ForceConvergence: false,
    InPlace: false,
    LabelsOnly: false,
    MaxIterations: 1000,
    Radius: 0,
    Verbose: false,
  }
}

/*
  This program performs mean shift clustering on the given dataset, storing the
  learned cluster assignments either as a column of labels in the input dataset
  or separately.
  
  The input dataset should be specified with the "Input" parameter, and the
  radius used for search can be specified with the "Radius" parameter.  The
  maximum number of iterations before algorithm termination is controlled with
  the "MaxIterations" parameter.
  
  The output labels may be saved with the "Output" output parameter and the
  centroids of each cluster may be saved with the "Centroid" output parameter.

  For example, to run mean shift clustering on the dataset data and store the
  centroids to centroids, the following command may be used: 
  
  // Initialize optional parameters for MeanShift().
  param := mlpack.MeanShiftOptions()
  
  centroids, _ := mlpack.MeanShift(data, param)

  Input parameters:

   - input (mat.Dense): Input dataset to perform clustering on.
   - ForceConvergence (bool): If specified, the mean shift algorithm will
        continue running regardless of max_iterations until the clusters
        converge.
   - InPlace (bool): If specified, a column containing the learned cluster
        assignments will be added to the input dataset file.  In this case,
        --output_file is overridden.  (Do not use with Python.)
   - LabelsOnly (bool): If specified, only the output labels will be
        written to the file specified by --output_file.
   - MaxIterations (int): Maximum number of iterations before mean shift
        terminates.  Default value 1000.
   - Radius (float64): If the distance between two centroids is less than
        the given radius, one will be removed.  A radius of 0 or less means an
        estimate will be calculated and used for the radius.  Default value 0.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - centroid (mat.Dense): If specified, the centroids of each cluster
        will be written to the given matrix.
   - output (mat.Dense): Matrix to write output labels or labeled data
        to.

 */
func MeanShift(input *mat.Dense, param *MeanShiftOptionalParam) (*mat.Dense, *mat.Dense) {
  params := getParams("mean_shift")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  gonumToArmaMat(params, "input", input, false)
  setPassed(params, "input")

  // Detect if the parameter was passed; set if so.
  if param.ForceConvergence != false {
    setParamBool(params, "force_convergence", param.ForceConvergence)
    setPassed(params, "force_convergence")
  }

  // Detect if the parameter was passed; set if so.
  if param.InPlace != false {
    setParamBool(params, "in_place", param.InPlace)
    setPassed(params, "in_place")
  }

  // Detect if the parameter was passed; set if so.
  if param.LabelsOnly != false {
    setParamBool(params, "labels_only", param.LabelsOnly)
    setPassed(params, "labels_only")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 1000 {
    setParamInt(params, "max_iterations", param.MaxIterations)
    setPassed(params, "max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Radius != 0 {
    setParamDouble(params, "radius", param.Radius)
    setPassed(params, "radius")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "centroid")
  setPassed(params, "output")

  // Call the mlpack program.
  C.mlpackMeanShift(params.mem, timers.mem)

  // Initialize result variable and get output.
  var centroidPtr mlpackArma
  centroid := centroidPtr.armaToGonumMat(params, "centroid")
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat(params, "output")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return centroid, output
}
