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
  params := getParams("knn")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "dual_tree" {
    setParamString(params, "algorithm", param.Algorithm)
    setPassed(params, "algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.Epsilon != 0 {
    setParamDouble(params, "epsilon", param.Epsilon)
    setPassed(params, "epsilon")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setKNNModel(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 0 {
    setParamInt(params, "k", param.K)
    setPassed(params, "k")
  }

  // Detect if the parameter was passed; set if so.
  if param.LeafSize != 20 {
    setParamInt(params, "leaf_size", param.LeafSize)
    setPassed(params, "leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Query != nil {
    gonumToArmaMat(params, "query", param.Query, false)
    setPassed(params, "query")
  }

  // Detect if the parameter was passed; set if so.
  if param.RandomBasis != false {
    setParamBool(params, "random_basis", param.RandomBasis)
    setPassed(params, "random_basis")
  }

  // Detect if the parameter was passed; set if so.
  if param.Reference != nil {
    gonumToArmaMat(params, "reference", param.Reference, false)
    setPassed(params, "reference")
  }

  // Detect if the parameter was passed; set if so.
  if param.Rho != 0.7 {
    setParamDouble(params, "rho", param.Rho)
    setPassed(params, "rho")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt(params, "seed", param.Seed)
    setPassed(params, "seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tau != 0 {
    setParamDouble(params, "tau", param.Tau)
    setPassed(params, "tau")
  }

  // Detect if the parameter was passed; set if so.
  if param.TreeType != "kd" {
    setParamString(params, "tree_type", param.TreeType)
    setPassed(params, "tree_type")
  }

  // Detect if the parameter was passed; set if so.
  if param.TrueDistances != nil {
    gonumToArmaMat(params, "true_distances", param.TrueDistances, false)
    setPassed(params, "true_distances")
  }

  // Detect if the parameter was passed; set if so.
  if param.TrueNeighbors != nil {
    gonumToArmaUmat(params, "true_neighbors", param.TrueNeighbors)
    setPassed(params, "true_neighbors")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "distances")
  setPassed(params, "neighbors")
  setPassed(params, "output_model")

  // Call the mlpack program.
  C.mlpackKnn(params.mem, timers.mem)

  // Initialize result variable and get output.
  var distancesPtr mlpackArma
  distances := distancesPtr.armaToGonumMat(params, "distances")
  var neighborsPtr mlpackArma
  neighbors := neighborsPtr.armaToGonumUmat(params, "neighbors")
  var outputModel knnModel
  outputModel.getKNNModel(params, "output_model")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return distances, neighbors, outputModel
}
