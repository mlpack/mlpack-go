package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_nmf
#include <capi/nmf.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type NmfOptionalParam struct {
    InitialH *mat.Dense
    InitialW *mat.Dense
    MaxIterations int
    MinResidue float64
    Seed int
    UpdateRules string
    Verbose bool
}

func NmfOptions() *NmfOptionalParam {
  return &NmfOptionalParam{
    InitialH: nil,
    InitialW: nil,
    MaxIterations: 10000,
    MinResidue: 1e-05,
    Seed: 0,
    UpdateRules: "multdist",
    Verbose: false,
  }
}

/*
  This program performs non-negative matrix factorization on the given dataset,
  storing the resulting decomposed matrices in the specified files.  For an
  input dataset V, NMF decomposes V into two matrices W and H such that 
  
  V = W * H
  
  where all elements in W and H are non-negative.  If V is of size (n x m), then
  W will be of size (n x r) and H will be of size (r x m), where r is the rank
  of the factorization (specified by the "Rank" parameter).
  
  Optionally, the desired update rules for each NMF iteration can be chosen from
  the following list:
  
   - multdist: multiplicative distance-based update rules (Lee and Seung 1999)
   - multdiv: multiplicative divergence-based update rules (Lee and Seung 1999)
   - als: alternating least squares update rules (Paatero and Tapper 1994)
  
  The maximum number of iterations is specified with "MaxIterations", and the
  minimum residue required for algorithm termination is specified with the
  "MinResidue" parameter.

  For example, to run NMF on the input matrix V using the 'multdist' update
  rules with a rank-10 decomposition and storing the decomposed matrices into W
  and H, the following command could be used: 
  
  // Initialize optional parameters for Nmf().
  param := mlpack.NmfOptions()
  param.UpdateRules = "multdist"
  
  H, W := mlpack.Nmf(V, 10, param)

  Input parameters:

   - input (mat.Dense): Input dataset to perform NMF on.
   - rank (int): Rank of the factorization.
   - InitialH (mat.Dense): Initial H matrix.
   - InitialW (mat.Dense): Initial W matrix.
   - MaxIterations (int): Number of iterations before NMF terminates (0
        runs until convergence.  Default value 10000.
   - MinResidue (float64): The minimum root mean square residue allowed
        for each iteration, below which the program terminates.  Default value
        1e-05.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - UpdateRules (string): Update rules for each iteration; ( multdist |
        multdiv | als ).  Default value 'multdist'.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - h (mat.Dense): Matrix to save the calculated H to.
   - w (mat.Dense): Matrix to save the calculated W to.

 */
func Nmf(input *mat.Dense, rank int, param *NmfOptionalParam) (*mat.Dense, *mat.Dense) {
  params := getParams("nmf")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  gonumToArmaMat(params, "input", input)
  setPassed(params, "input")

  // Detect if the parameter was passed; set if so.
  setParamInt(params, "rank", rank)
  setPassed(params, "rank")

  // Detect if the parameter was passed; set if so.
  if param.InitialH != nil {
    gonumToArmaMat(params, "initial_h", param.InitialH)
    setPassed(params, "initial_h")
  }

  // Detect if the parameter was passed; set if so.
  if param.InitialW != nil {
    gonumToArmaMat(params, "initial_w", param.InitialW)
    setPassed(params, "initial_w")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 10000 {
    setParamInt(params, "max_iterations", param.MaxIterations)
    setPassed(params, "max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinResidue != 1e-05 {
    setParamDouble(params, "min_residue", param.MinResidue)
    setPassed(params, "min_residue")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt(params, "seed", param.Seed)
    setPassed(params, "seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.UpdateRules != "multdist" {
    setParamString(params, "update_rules", param.UpdateRules)
    setPassed(params, "update_rules")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "h")
  setPassed(params, "w")

  // Call the mlpack program.
  C.mlpackNmf(params.mem, timers.mem)

  // Initialize result variable and get output.
  var hPtr mlpackArma
  h := hPtr.armaToGonumMat(params, "h")
  var wPtr mlpackArma
  w := wPtr.armaToGonumMat(params, "w")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return h, w
}
