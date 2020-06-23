package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_kfn
#include <capi/kfn.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type KfnOptionalParam struct {
    Algorithm string
    Epsilon float64
    InputModel *kfnModel
    K int
    LeafSize int
    Percentage float64
    Query *mat.Dense
    RandomBasis bool
    Reference *mat.Dense
    Seed int
    TreeType string
    TrueDistances *mat.Dense
    TrueNeighbors *mat.Dense
    Verbose bool
}

func KfnOptions() *KfnOptionalParam {
  return &KfnOptionalParam{
    Algorithm: "dual_tree",
    Epsilon: 0,
    InputModel: nil,
    K: 0,
    LeafSize: 20,
    Percentage: 1,
    Query: nil,
    RandomBasis: false,
    Reference: nil,
    Seed: 0,
    TreeType: "kd",
    TrueDistances: nil,
    TrueNeighbors: nil,
    Verbose: false,
  }
}

type kfnModel struct {
  mem unsafe.Pointer
}

func (m *kfnModel) allocKFNModel(identifier string) {
  m.mem = C.mlpackGetKFNModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *kfnModel) getKFNModel(identifier string) {
  m.allocKFNModel(identifier)
}

func setKFNModel(identifier string, ptr *kfnModel) {
  C.mlpackSetKFNModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program will calculate the k-furthest-neighbors of a set of points. You
  may specify a separate set of reference points and query points, or just a
  reference set which will be used as both the reference and query set.
  
  For example, the following will calculate the 5 furthest neighbors of
  eachpoint in input and store the distances in distances and the neighbors in
  neighbors: 
  
  // Initialize optional parameters for Kfn().
  param := mlpack.KfnOptions()
  param.K = 5
  param.Reference = input
  
  distances, neighbors, _ := mlpack.Kfn(param)
  
  The output files are organized such that row i and column j in the neighbors
  output matrix corresponds to the index of the point in the reference set which
  is the j'th furthest neighbor from the point in the query set with index i. 
  Row i and column j in the distances output file corresponds to the distance
  between those two points.


  Input parameters:

   - Algorithm (string): Type of neighbor search: 'naive', 'single_tree',
        'dual_tree', 'greedy'.  Default value 'dual_tree'.
   - Epsilon (float64): If specified, will do approximate furthest
        neighbor search with given relative error. Must be in the range [0,1). 
        Default value 0.
   - InputModel (kfnModel): Pre-trained kFN model.
   - K (int): Number of furthest neighbors to find.  Default value 0.
   - LeafSize (int): Leaf size for tree building (used for kd-trees, vp
        trees, random projection trees, UB trees, R trees, R* trees, X trees,
        Hilbert R trees, R+ trees, R++ trees, and octrees).  Default value 20.
   - Percentage (float64): If specified, will do approximate furthest
        neighbor search. Must be in the range (0,1] (decimal form). Resultant
        neighbors will be at least (p*100) % of the distance as the true
        furthest neighbor.  Default value 1.
   - Query (mat.Dense): Matrix containing query points (optional).
   - RandomBasis (bool): Before tree-building, project the data onto a
        random orthogonal basis.
   - Reference (mat.Dense): Matrix containing the reference dataset.
   - Seed (int): Random seed (if 0, std::time(NULL) is used).  Default
        value 0.
   - TreeType (string): Type of tree to use: 'kd', 'vp', 'rp', 'max-rp',
        'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus',
        'r-plus-plus', 'oct'.  Default value 'kd'.
   - TrueDistances (mat.Dense): Matrix of true distances to compute the
        effective error (average relative error) (it is printed when -v is
        specified).
   - TrueNeighbors (mat.Dense): Matrix of true neighbors to compute the
        recall (it is printed when -v is specified).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - distances (mat.Dense): Matrix to output distances into.
   - neighbors (mat.Dense): Matrix to output neighbors into.
   - outputModel (kfnModel): If specified, the kFN model will be output
        here.

 */
func Kfn(param *KfnOptionalParam) (*mat.Dense, *mat.Dense, kfnModel) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("k-Furthest-Neighbors Search")

  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "dual_tree" {
    setParamString("algorithm", param.Algorithm)
    setPassed("algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.Epsilon != 0 {
    setParamDouble("epsilon", param.Epsilon)
    setPassed("epsilon")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setKFNModel("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 0 {
    setParamInt("k", param.K)
    setPassed("k")
  }

  // Detect if the parameter was passed; set if so.
  if param.LeafSize != 20 {
    setParamInt("leaf_size", param.LeafSize)
    setPassed("leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Percentage != 1 {
    setParamDouble("percentage", param.Percentage)
    setPassed("percentage")
  }

  // Detect if the parameter was passed; set if so.
  if param.Query != nil {
    gonumToArmaMat("query", param.Query)
    setPassed("query")
  }

  // Detect if the parameter was passed; set if so.
  if param.RandomBasis != false {
    setParamBool("random_basis", param.RandomBasis)
    setPassed("random_basis")
  }

  // Detect if the parameter was passed; set if so.
  if param.Reference != nil {
    gonumToArmaMat("reference", param.Reference)
    setPassed("reference")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.TreeType != "kd" {
    setParamString("tree_type", param.TreeType)
    setPassed("tree_type")
  }

  // Detect if the parameter was passed; set if so.
  if param.TrueDistances != nil {
    gonumToArmaMat("true_distances", param.TrueDistances)
    setPassed("true_distances")
  }

  // Detect if the parameter was passed; set if so.
  if param.TrueNeighbors != nil {
    gonumToArmaUmat("true_neighbors", param.TrueNeighbors)
    setPassed("true_neighbors")
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
  C.mlpackKfn()

  // Initialize result variable and get output.
  var distancesPtr mlpackArma
  distances := distancesPtr.armaToGonumMat("distances")
  var neighborsPtr mlpackArma
  neighbors := neighborsPtr.armaToGonumUmat("neighbors")
  var outputModel kfnModel
  outputModel.getKFNModel("output_model")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return distances, neighbors, outputModel
}
