package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_approx_kfn
#include <capi/approx_kfn.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type ApproxKfnOptionalParam struct {
    Algorithm string
    CalculateError bool
    ExactDistances *mat.Dense
    InputModel *approxkfnModel
    K int
    NumProjections int
    NumTables int
    Query *mat.Dense
    Reference *mat.Dense
    Verbose bool
}

func ApproxKfnOptions() *ApproxKfnOptionalParam {
  return &ApproxKfnOptionalParam{
    Algorithm: "ds",
    CalculateError: false,
    ExactDistances: nil,
    InputModel: nil,
    K: 0,
    NumProjections: 5,
    NumTables: 5,
    Query: nil,
    Reference: nil,
    Verbose: false,
  }
}

/*
  This program implements two strategies for furthest neighbor search. These
  strategies are:
  
   - The 'qdafn' algorithm from "Approximate Furthest Neighbor in High
  Dimensions" by R. Pagh, F. Silvestri, J. Sivertsen, and M. Skala, in
  Similarity Search and Applications 2015 (SISAP).
   - The 'DrusillaSelect' algorithm from "Fast approximate furthest neighbors
  with data-dependent candidate selection", by R.R. Curtin and A.B. Gardner, in
  Similarity Search and Applications 2016 (SISAP).
  
  These two strategies give approximate results for the furthest neighbor search
  problem and can be used as fast replacements for other furthest neighbor
  techniques such as those found in the mlpack_kfn program.  Note that
  typically, the 'ds' algorithm requires far fewer tables and projections than
  the 'qdafn' algorithm.
  
  Specify a reference set (set to search in) with "Reference", specify a query
  set with "Query", and specify algorithm parameters with "NumTables" and
  "NumProjections" (or don't and defaults will be used).  The algorithm to be
  used (either 'ds'---the default---or 'qdafn')  may be specified with
  "Algorithm".  Also specify the number of neighbors to search for with "K".
  
  Note that for 'qdafn' in lower dimensions, "NumProjections" may need to be set
  to a high value in order to return results for each query point.
  
  If no query set is specified, the reference set will be used as the query set.
   The "OutputModel" output parameter may be used to store the built model, and
  an input model may be loaded instead of specifying a reference set with the
  "InputModel" option.
  
  Results for each query point can be stored with the "Neighbors" and
  "Distances" output parameters.  Each row of these output matrices holds the k
  distances or neighbor indices for each query point.

  For example, to find the 5 approximate furthest neighbors with reference_set
  as the reference set and query_set as the query set using DrusillaSelect,
  storing the furthest neighbor indices to neighbors and the furthest neighbor
  distances to distances, one could call
  
  // Initialize optional parameters for ApproxKfn().
  param := mlpack.ApproxKfnOptions()
  param.Query = query_set
  param.Reference = reference_set
  param.K = 5
  param.Algorithm = "ds"
  
  distances, neighbors, _ := mlpack.ApproxKfn(param)
  
  and to perform approximate all-furthest-neighbors search with k=1 on the set
  data storing only the furthest neighbor distances to distances, one could call
  
  // Initialize optional parameters for ApproxKfn().
  param := mlpack.ApproxKfnOptions()
  param.Reference = reference_set
  param.K = 1
  
  distances, _, _ := mlpack.ApproxKfn(param)
  
  A trained model can be re-used.  If a model has been previously saved to
  model, then we may find 3 approximate furthest neighbors on a query set
  new_query_set using that model and store the furthest neighbor indices into
  neighbors by calling
  
  // Initialize optional parameters for ApproxKfn().
  param := mlpack.ApproxKfnOptions()
  param.InputModel = &model
  param.Query = new_query_set
  param.K = 3
  
  _, neighbors, _ := mlpack.ApproxKfn(param)

  Input parameters:

   - Algorithm (string): Algorithm to use: 'ds' or 'qdafn'.  Default value
        'ds'.
   - CalculateError (bool): If set, calculate the average distance error
        for the first furthest neighbor only.
   - ExactDistances (mat.Dense): Matrix containing exact distances to
        furthest neighbors; this can be used to avoid explicit calculation when
        --calculate_error is set.
   - InputModel (approxkfnModel): File containing input model.
   - K (int): Number of furthest neighbors to search for.  Default value
        0.
   - NumProjections (int): Number of projections to use in each hash
        table.  Default value 5.
   - NumTables (int): Number of hash tables to use.  Default value 5.
   - Query (mat.Dense): Matrix containing query points.
   - Reference (mat.Dense): Matrix containing the reference dataset.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - distances (mat.Dense): Matrix to save furthest neighbor distances
        to.
   - neighbors (mat.Dense): Matrix to save neighbor indices to.
   - outputModel (approxkfnModel): File to save output model to.

 */
func ApproxKfn(param *ApproxKfnOptionalParam) (*mat.Dense, *mat.Dense, approxkfnModel) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Approximate furthest neighbor search")

  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "ds" {
    setParamString("algorithm", param.Algorithm)
    setPassed("algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.CalculateError != false {
    setParamBool("calculate_error", param.CalculateError)
    setPassed("calculate_error")
  }

  // Detect if the parameter was passed; set if so.
  if param.ExactDistances != nil {
    gonumToArmaMat("exact_distances", param.ExactDistances)
    setPassed("exact_distances")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setApproxKFNModel("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 0 {
    setParamInt("k", param.K)
    setPassed("k")
  }

  // Detect if the parameter was passed; set if so.
  if param.NumProjections != 5 {
    setParamInt("num_projections", param.NumProjections)
    setPassed("num_projections")
  }

  // Detect if the parameter was passed; set if so.
  if param.NumTables != 5 {
    setParamInt("num_tables", param.NumTables)
    setPassed("num_tables")
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
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("distances")
  setPassed("neighbors")
  setPassed("output_model")

  // Call the mlpack program.
  C.mlpackApproxKfn()

  // Initialize result variable and get output.
  var distancesPtr mlpackArma
  distances := distancesPtr.armaToGonumMat("distances")
  var neighborsPtr mlpackArma
  neighbors := neighborsPtr.armaToGonumUmat("neighbors")
  var outputModel approxkfnModel
  outputModel.getApproxKFNModel("output_model")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return distances, neighbors, outputModel
}
