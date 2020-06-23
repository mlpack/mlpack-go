package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_dbscan
#include <capi/dbscan.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type DbscanOptionalParam struct {
    Epsilon float64
    MinSize int
    Naive bool
    SelectionType string
    SingleMode bool
    TreeType string
    Verbose bool
}

func DbscanOptions() *DbscanOptionalParam {
  return &DbscanOptionalParam{
    Epsilon: 1,
    MinSize: 5,
    Naive: false,
    SelectionType: "ordered",
    SingleMode: false,
    TreeType: "kd",
    Verbose: false,
  }
}

/*
  This program implements the DBSCAN algorithm for clustering using accelerated
  tree-based range search.  The type of tree that is used may be parameterized,
  or brute-force range search may also be used.
  
  The input dataset to be clustered may be specified with the "Input" parameter;
  the radius of each range search may be specified with the "Epsilon"
  parameters, and the minimum number of points in a cluster may be specified
  with the "MinSize" parameter.
  
  The "Assignments" and "Centroids" output parameters may be used to save the
  output of the clustering. "Assignments" contains the cluster assignments of
  each point, and "Centroids" contains the centroids of each cluster.
  
  The range search may be controlled with the "TreeType", "SingleMode", and
  "Naive" parameters.  "TreeType" can control the type of tree used for range
  search; this can take a variety of values: 'kd', 'r', 'r-star', 'x',
  'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'. The "SingleMode"
  parameter will force single-tree search (as opposed to the default dual-tree
  search), and '"Naive" will force brute-force range search.
  
  An example usage to run DBSCAN on the dataset in input with a radius of 0.5
  and a minimum cluster size of 5 is given below:
  
  // Initialize optional parameters for Dbscan().
  param := mlpack.DbscanOptions()
  param.Epsilon = 0.5
  param.MinSize = 5
  
  _, _ := mlpack.Dbscan(input, param)


  Input parameters:

   - input (mat.Dense): Input dataset to cluster.
   - Epsilon (float64): Radius of each range search.  Default value 1.
   - MinSize (int): Minimum number of points for a cluster.  Default value
        5.
   - Naive (bool): If set, brute-force range search (not tree-based) will
        be used.
   - SelectionType (string): If using point selection policy, the type of
        selection to use ('ordered', 'random').  Default value 'ordered'.
   - SingleMode (bool): If set, single-tree range search (not dual-tree)
        will be used.
   - TreeType (string): If using single-tree or dual-tree search, the type
        of tree to use ('kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus',
        'r-plus-plus', 'cover', 'ball').  Default value 'kd'.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - assignments (mat.Dense): Output matrix for assignments of each
        point.
   - centroids (mat.Dense): Matrix to save output centroids to.

 */
func Dbscan(input *mat.Dense, param *DbscanOptionalParam) (*mat.Dense, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("DBSCAN clustering")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Epsilon != 1 {
    setParamDouble("epsilon", param.Epsilon)
    setPassed("epsilon")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinSize != 5 {
    setParamInt("min_size", param.MinSize)
    setPassed("min_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Naive != false {
    setParamBool("naive", param.Naive)
    setPassed("naive")
  }

  // Detect if the parameter was passed; set if so.
  if param.SelectionType != "ordered" {
    setParamString("selection_type", param.SelectionType)
    setPassed("selection_type")
  }

  // Detect if the parameter was passed; set if so.
  if param.SingleMode != false {
    setParamBool("single_mode", param.SingleMode)
    setPassed("single_mode")
  }

  // Detect if the parameter was passed; set if so.
  if param.TreeType != "kd" {
    setParamString("tree_type", param.TreeType)
    setPassed("tree_type")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("assignments")
  setPassed("centroids")

  // Call the mlpack program.
  C.mlpackDbscan()

  // Initialize result variable and get output.
  var assignmentsPtr mlpackArma
  assignments := assignmentsPtr.armaToGonumUrow("assignments")
  var centroidsPtr mlpackArma
  centroids := centroidsPtr.armaToGonumMat("centroids")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return assignments, centroids
}
