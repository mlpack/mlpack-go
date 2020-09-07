package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_emst
#include <capi/emst.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type EmstOptionalParam struct {
    LeafSize int
    Naive bool
    Verbose bool
}

func EmstOptions() *EmstOptionalParam {
  return &EmstOptionalParam{
    LeafSize: 1,
    Naive: false,
    Verbose: false,
  }
}

/*
  This program can compute the Euclidean minimum spanning tree of a set of input
  points using the dual-tree Boruvka algorithm.
  
  The set to calculate the minimum spanning tree of is specified with the
  "Input" parameter, and the output may be saved with the "Output" output
  parameter.
  
  The "LeafSize" parameter controls the leaf size of the kd-tree that is used to
  calculate the minimum spanning tree, and if the "Naive" option is given, then
  brute-force search is used (this is typically much slower in low dimensions). 
  The leaf size does not affect the results, but it may have some effect on the
  runtime of the algorithm.

  For example, the minimum spanning tree of the input dataset data can be
  calculated with a leaf size of 20 and stored as spanning_tree using the
  following command:
  
  // Initialize optional parameters for Emst().
  param := mlpack.EmstOptions()
  param.LeafSize = 20
  
  spanning_tree := mlpack.Emst(data, param)
  
  The output matrix is a three-dimensional matrix, where each row indicates an
  edge.  The first dimension corresponds to the lesser index of the edge; the
  second dimension corresponds to the greater index of the edge; and the third
  column corresponds to the distance between the two points.

  Input parameters:

   - input (mat.Dense): Input data matrix.
   - LeafSize (int): Leaf size in the kd-tree.  One-element leaves give
        the empirically best performance, but at the cost of greater memory
        requirements.  Default value 1.
   - Naive (bool): Compute the MST using O(n^2) naive algorithm.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Output data.  Stored as an edge list.

 */
func Emst(input *mat.Dense, param *EmstOptionalParam) (*mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Fast Euclidean Minimum Spanning Tree")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.LeafSize != 1 {
    setParamInt("leaf_size", param.LeafSize)
    setPassed("leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Naive != false {
    setParamBool("naive", param.Naive)
    setPassed("naive")
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
  C.mlpackEmst()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat("output")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return output
}
