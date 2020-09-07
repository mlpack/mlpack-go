package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_lsh
#include <capi/lsh.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type LshOptionalParam struct {
    BucketSize int
    HashWidth float64
    InputModel *lshSearch
    K int
    NumProbes int
    Projections int
    Query *mat.Dense
    Reference *mat.Dense
    SecondHashSize int
    Seed int
    Tables int
    TrueNeighbors *mat.Dense
    Verbose bool
}

func LshOptions() *LshOptionalParam {
  return &LshOptionalParam{
    BucketSize: 500,
    HashWidth: 0,
    InputModel: nil,
    K: 0,
    NumProbes: 0,
    Projections: 10,
    Query: nil,
    Reference: nil,
    SecondHashSize: 99901,
    Seed: 0,
    Tables: 30,
    TrueNeighbors: nil,
    Verbose: false,
  }
}

/*
  This program will calculate the k approximate-nearest-neighbors of a set of
  points using locality-sensitive hashing. You may specify a separate set of
  reference points and query points, or just a reference set which will be used
  as both the reference and query set. 

  For example, the following will return 5 neighbors from the data for each
  point in input and store the distances in distances and the neighbors in
  neighbors:
  
  // Initialize optional parameters for Lsh().
  param := mlpack.LshOptions()
  param.K = 5
  param.Reference = input
  
  distances, neighbors, _ := mlpack.Lsh(param)
  
  The output is organized such that row i and column j in the neighbors output
  corresponds to the index of the point in the reference set which is the j'th
  nearest neighbor from the point in the query set with index i.  Row j and
  column i in the distances output file corresponds to the distance between
  those two points.
  
  Because this is approximate-nearest-neighbors search, results may be different
  from run to run.  Thus, the "Seed" parameter can be specified to set the
  random seed.
  
  This program also has many other parameters to control its functionality; see
  the parameter-specific documentation for more information.

  Input parameters:

   - BucketSize (int): The size of a bucket in the second level hash. 
        Default value 500.
   - HashWidth (float64): The hash width for the first-level hashing in
        the LSH preprocessing. By default, the LSH class automatically estimates
        a hash width for its use.  Default value 0.
   - InputModel (lshSearch): Input LSH model.
   - K (int): Number of nearest neighbors to find.  Default value 0.
   - NumProbes (int): Number of additional probes for multiprobe LSH; if
        0, traditional LSH is used.  Default value 0.
   - Projections (int): The number of hash functions for each table 
        Default value 10.
   - Query (mat.Dense): Matrix containing query points (optional).
   - Reference (mat.Dense): Matrix containing the reference dataset.
   - SecondHashSize (int): The size of the second level hash table. 
        Default value 99901.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - Tables (int): The number of hash tables to be used.  Default value
        30.
   - TrueNeighbors (mat.Dense): Matrix of true neighbors to compute recall
        with (the recall is printed when -v is specified).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - distances (mat.Dense): Matrix to output distances into.
   - neighbors (mat.Dense): Matrix to output neighbors into.
   - outputModel (lshSearch): Output for trained LSH model.

 */
func Lsh(param *LshOptionalParam) (*mat.Dense, *mat.Dense, lshSearch) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("K-Approximate-Nearest-Neighbor Search with LSH")

  // Detect if the parameter was passed; set if so.
  if param.BucketSize != 500 {
    setParamInt("bucket_size", param.BucketSize)
    setPassed("bucket_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.HashWidth != 0 {
    setParamDouble("hash_width", param.HashWidth)
    setPassed("hash_width")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setLSHSearch("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.K != 0 {
    setParamInt("k", param.K)
    setPassed("k")
  }

  // Detect if the parameter was passed; set if so.
  if param.NumProbes != 0 {
    setParamInt("num_probes", param.NumProbes)
    setPassed("num_probes")
  }

  // Detect if the parameter was passed; set if so.
  if param.Projections != 10 {
    setParamInt("projections", param.Projections)
    setPassed("projections")
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
  if param.SecondHashSize != 99901 {
    setParamInt("second_hash_size", param.SecondHashSize)
    setPassed("second_hash_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tables != 30 {
    setParamInt("tables", param.Tables)
    setPassed("tables")
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
  C.mlpackLsh()

  // Initialize result variable and get output.
  var distancesPtr mlpackArma
  distances := distancesPtr.armaToGonumMat("distances")
  var neighborsPtr mlpackArma
  neighbors := neighborsPtr.armaToGonumUmat("neighbors")
  var outputModel lshSearch
  outputModel.getLSHSearch("output_model")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return distances, neighbors, outputModel
}
