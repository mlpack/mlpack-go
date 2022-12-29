package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_krann
#include <capi/krann.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type KrannOptionalParam struct {
    Alpha float64
    FirstLeafExact bool
    InputModel *raModel
    K int
    LeafSize int
    Naive bool
    Query *mat.Dense
    RandomBasis bool
    Reference *mat.Dense
    SampleAtLeaves bool
    Seed int
    SingleMode bool
    SingleSampleLimit int
    Tau float64
    TreeType string
    Verbose bool
}

func KrannOptions() *KrannOptionalParam {
  return &KrannOptionalParam{
    Alpha: 0.95,
    FirstLeafExact: false,
    InputModel: nil,
    K: 0,
    LeafSize: 20,
    Naive: false,
    Query: nil,
    RandomBasis: false,
    Reference: nil,
    SampleAtLeaves: false,
    Seed: 0,
    SingleMode: false,
    SingleSampleLimit: 20,
    Tau: 5,
    TreeType: "kd",
    Verbose: false,
  }
}

/*
  This program will calculate the k rank-approximate-nearest-neighbors of a set
  of points. You may specify a separate set of reference points and query
  points, or just a reference set which will be used as both the reference and
  query set. You must specify the rank approximation (in %) (and optionally the
  success probability).

  For example, the following will return 5 neighbors from the top 0.1% of the
  data (with probability 0.95) for each point in input and store the distances
  in distances and the neighbors in neighbors.csv:
  
  // Initialize optional parameters for Krann().
  param := mlpack.KrannOptions()
  param.Reference = input
  param.K = 5
  param.Tau = 0.1
  
  distances, neighbors, _ := mlpack.Krann(param)
  
  Note that tau must be set such that the number of points in the corresponding
  percentile of the data is greater than k.  Thus, if we choose tau = 0.1 with a
  dataset of 1000 points and k = 5, then we are attempting to choose 5 nearest
  neighbors out of the closest 1 point -- this is invalid and the program will
  terminate with an error message.
  
  The output matrices are organized such that row i and column j in the
  neighbors output file corresponds to the index of the point in the reference
  set which is the i'th nearest neighbor from the point in the query set with
  index j.  Row i and column j in the distances output file corresponds to the
  distance between those two points.

  Input parameters:

   - Alpha (float64): The desired success probability.  Default value
        0.95.
   - FirstLeafExact (bool): The flag to trigger sampling only after
        exactly exploring the first leaf.
   - InputModel (raModel): Pre-trained kNN model.
   - K (int): Number of nearest neighbors to find.  Default value 0.
   - LeafSize (int): Leaf size for tree building (used for kd-trees, UB
        trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees,
        and octrees).  Default value 20.
   - Naive (bool): If true, sampling will be done without using a tree.
   - Query (mat.Dense): Matrix containing query points (optional).
   - RandomBasis (bool): Before tree-building, project the data onto a
        random orthogonal basis.
   - Reference (mat.Dense): Matrix containing the reference dataset.
   - SampleAtLeaves (bool): The flag to trigger sampling at leaves.
   - Seed (int): Random seed (if 0, std::time(NULL) is used).  Default
        value 0.
   - SingleMode (bool): If true, single-tree search is used (as opposed to
        dual-tree search.
   - SingleSampleLimit (int): The limit on the maximum number of samples
        (and hence the largest node you can approximate).  Default value 20.
   - Tau (float64): The allowed rank-error in terms of the percentile of
        the data.  Default value 5.
   - TreeType (string): Type of tree to use: 'kd', 'ub', 'cover', 'r',
        'x', 'r-star', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'.  Default
        value 'kd'.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - distances (mat.Dense): Matrix to output distances into.
   - neighbors (mat.Dense): Matrix to output neighbors into.
   - outputModel (raModel): If specified, the kNN model will be output
        here.

 */
func Krann(param *KrannOptionalParam) (*mat.Dense, *mat.Dense, raModel) {
  params := getParams("krann")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.Alpha != 0.95 {
    setParamDouble(params, "alpha", param.Alpha)
    setPassed(params, "alpha")
  }

  // Detect if the parameter was passed; set if so.
  if param.FirstLeafExact != false {
    setParamBool(params, "first_leaf_exact", param.FirstLeafExact)
    setPassed(params, "first_leaf_exact")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setRAModel(params, "input_model", param.InputModel)
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
  if param.Naive != false {
    setParamBool(params, "naive", param.Naive)
    setPassed(params, "naive")
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
  if param.SampleAtLeaves != false {
    setParamBool(params, "sample_at_leaves", param.SampleAtLeaves)
    setPassed(params, "sample_at_leaves")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt(params, "seed", param.Seed)
    setPassed(params, "seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.SingleMode != false {
    setParamBool(params, "single_mode", param.SingleMode)
    setPassed(params, "single_mode")
  }

  // Detect if the parameter was passed; set if so.
  if param.SingleSampleLimit != 20 {
    setParamInt(params, "single_sample_limit", param.SingleSampleLimit)
    setPassed(params, "single_sample_limit")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tau != 5 {
    setParamDouble(params, "tau", param.Tau)
    setPassed(params, "tau")
  }

  // Detect if the parameter was passed; set if so.
  if param.TreeType != "kd" {
    setParamString(params, "tree_type", param.TreeType)
    setPassed(params, "tree_type")
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
  C.mlpackKrann(params.mem, timers.mem)

  // Initialize result variable and get output.
  var distancesPtr mlpackArma
  distances := distancesPtr.armaToGonumMat(params, "distances")
  var neighborsPtr mlpackArma
  neighbors := neighborsPtr.armaToGonumUmat(params, "neighbors")
  var outputModel raModel
  outputModel.getRAModel(params, "output_model")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return distances, neighbors, outputModel
}
