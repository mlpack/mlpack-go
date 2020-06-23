package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_radical
#include <capi/radical.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type RadicalOptionalParam struct {
    Angles int
    NoiseStdDev float64
    Objective bool
    Replicates int
    Seed int
    Sweeps int
    Verbose bool
}

func RadicalOptions() *RadicalOptionalParam {
  return &RadicalOptionalParam{
    Angles: 150,
    NoiseStdDev: 0.175,
    Objective: false,
    Replicates: 30,
    Seed: 0,
    Sweeps: 0,
    Verbose: false,
  }
}

/*
  An implementation of RADICAL, a method for independent component analysis
  (ICA).  Assuming that we have an input matrix X, the goal is to find a square
  unmixing matrix W such that Y = W * X and the dimensions of Y are independent
  components.  If the algorithm is running particularly slowly, try reducing the
  number of replicates.
  
  The input matrix to perform ICA on should be specified with the "Input"
  parameter.  The output matrix Y may be saved with the "OutputIc" output
  parameter, and the output unmixing matrix W may be saved with the
  "OutputUnmixing" output parameter.
  
  For example, to perform ICA on the matrix X with 40 replicates, saving the
  independent components to ic, the following command may be used: 
  
  // Initialize optional parameters for Radical().
  param := mlpack.RadicalOptions()
  param.Replicates = 40
  
  ic, _ := mlpack.Radical(X, param)


  Input parameters:

   - input (mat.Dense): Input dataset for ICA.
   - Angles (int): Number of angles to consider in brute-force search
        during Radical2D.  Default value 150.
   - NoiseStdDev (float64): Standard deviation of Gaussian noise.  Default
        value 0.175.
   - Objective (bool): If set, an estimate of the final objective function
        is printed.
   - Replicates (int): Number of Gaussian-perturbed replicates to use (per
        point) in Radical2D.  Default value 30.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - Sweeps (int): Number of sweeps; each sweep calls Radical2D once for
        each pair of dimensions.  Default value 0.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputIc (mat.Dense): Matrix to save independent components to.
   - outputUnmixing (mat.Dense): Matrix to save unmixing matrix to.

 */
func Radical(input *mat.Dense, param *RadicalOptionalParam) (*mat.Dense, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("RADICAL")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Angles != 150 {
    setParamInt("angles", param.Angles)
    setPassed("angles")
  }

  // Detect if the parameter was passed; set if so.
  if param.NoiseStdDev != 0.175 {
    setParamDouble("noise_std_dev", param.NoiseStdDev)
    setPassed("noise_std_dev")
  }

  // Detect if the parameter was passed; set if so.
  if param.Objective != false {
    setParamBool("objective", param.Objective)
    setPassed("objective")
  }

  // Detect if the parameter was passed; set if so.
  if param.Replicates != 30 {
    setParamInt("replicates", param.Replicates)
    setPassed("replicates")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Sweeps != 0 {
    setParamInt("sweeps", param.Sweeps)
    setPassed("sweeps")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("output_ic")
  setPassed("output_unmixing")

  // Call the mlpack program.
  C.mlpackRadical()

  // Initialize result variable and get output.
  var outputIcPtr mlpackArma
  outputIc := outputIcPtr.armaToGonumMat("output_ic")
  var outputUnmixingPtr mlpackArma
  outputUnmixing := outputUnmixingPtr.armaToGonumMat("output_unmixing")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return outputIc, outputUnmixing
}
