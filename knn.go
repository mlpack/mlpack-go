package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_knn
#include <capi/knn.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type KnnOptionalParam struct {
    Algorithm string
    Epsilon float64
    InputModel *knnModel
    K int
    LeafSize int
    Query *mat.Dense
    RandomBasis bool
    Reference *mat.Dense
    Rho float64
    Seed int
    Tau float64
    TreeType string
    TrueDistances *mat.Dense
    TrueNeighbors *mat.Dense
    Verbose bool
}

func KnnOptions() *KnnOptionalParam {
  return &KnnOptionalParam{
    Algorithm: "dual_tree",
    Epsilon: 0,
    InputModel: nil,
    K: 0,
    LeafSize: 20,
    Query: nil,
    RandomBasis: false,
    Reference: nil,
    Rho: 0.7,
    Seed: 0,
    Tau: 0,
    TreeType: "kd",
    TrueDistances: nil,
    TrueNeighbors: nil,
    Verbose: false,
  }
}

/*
  This program will calculate the k-nearest-neighbors of a set of points using
  kd-trees or cover trees (cover tree support is experimental and may be slow).
  You may specify a separate set of reference points and query points, or just a
  reference set which will be used as both the reference and query set.

  For example, the following command will calculate the 5 nearest neighbors of
  each point in input and store the distances in distances and the neighbors in
  neighbors: 
  
  // Initialize optional parameters for Knn().
  param := mlpack.KnnOptions()
  param.K = 5
  param.Reference = input
  
  distances, neighbors, _ := mlpack.Knn(param)
  
  The output is organized such that row i and column j in the neighbors output
  matrix corresponds to the index of the point in the reference set which is the
  j'th nearest neighbor from the point in the query set with index i.  Row j and
  column i in the distances output matrix corresponds to the distance between
  those two points.

  Input parameters:

   - Algorithm (string): Type of neighbor search: 'naive', 'single_tree',
        'dual_tree', 'greedy'.  Default value 'dual_tree'.
   - Epsilon (float64): If specified, will do approximate nearest neighbor
        search with given relative error.  Default value 0.
   - InputModel (knnModel): Pre-trained kNN model.
   - K (int): Number of nearest neighbors to find.  Default value 0.
   - LeafSize (int): Leaf size for tree building (used for kd-trees, vp
        trees, random projection trees, UB trees, R trees, R* trees, X trees,
        Hilbert R trees, R+ trees, R++ trees, spill trees, and octrees). 
        Default value 20.
   - Query (mat.Dense): Matrix containing query points (optional).
   - RandomBasis (bool): Before tree-building, project the data onto a
        random orthogonal basis.
   - Reference (mat.Dense): Matrix containing the reference dataset.
   - Rho (float64): Balance threshold (only valid for spill trees). 
        Default value 0.7.
   - Seed (int): Random seed (if 0, std::time(NULL) is used).  Default
        value 0.
   - Tau (float64): Overlapping size (only valid for spill trees). 
        Default value 0.
   - TreeType (string): Type of tree to use: 'kd', 'vp', 'rp', 'max-rp',
        'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus',
        'r-plus-plus', 'spill', 'oct'.  Default value 'kd'.
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
   - outputModel (knnModel): If specified, the kNN model will be output
        here.

 */
func Knn(param *KnnOptionalParam) (*mat.Dense, *mat.Dense, knnModel) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("k-Nearest-Neighbors Search")

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
    setKNNModel("input_model", param.InputModel)
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
  if param.Rho != 0.7 {
    setParamDouble("rho", param.Rho)
    setPassed("rho")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tau != 0 {
    setParamDouble("tau", param.Tau)
    setPassed("tau")
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
  C.mlpackKnn()

  // Initialize result variable and get output.
  var distancesPtr mlpackArma
  distances := distancesPtr.armaToGonumMat("distances")
  var neighborsPtr mlpackArma
  neighbors := neighborsPtr.armaToGonumUmat("neighbors")
  var outputModel knnModel
  outputModel.getKNNModel("output_model")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return distances, neighbors, outputModel
}
