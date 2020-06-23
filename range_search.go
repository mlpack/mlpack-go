package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_range_search
#include <capi/range_search.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type RangeSearchOptionalParam struct {
    InputModel *rsModel
    LeafSize int
    Max float64
    Min float64
    Naive bool
    Query *mat.Dense
    RandomBasis bool
    Reference *mat.Dense
    Seed int
    SingleMode bool
    TreeType string
    Verbose bool
}

func RangeSearchOptions() *RangeSearchOptionalParam {
  return &RangeSearchOptionalParam{
    InputModel: nil,
    LeafSize: 20,
    Max: 0,
    Min: 0,
    Naive: false,
    Query: nil,
    RandomBasis: false,
    Reference: nil,
    Seed: 0,
    SingleMode: false,
    TreeType: "kd",
    Verbose: false,
  }
}

type rsModel struct {
  mem unsafe.Pointer
}

func (m *rsModel) allocRSModel(identifier string) {
  m.mem = C.mlpackGetRSModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *rsModel) getRSModel(identifier string) {
  m.allocRSModel(identifier)
}

func setRSModel(identifier string, ptr *rsModel) {
  C.mlpackSetRSModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program implements range search with a Euclidean distance metric. For a
  given query point, a given range, and a given set of reference points, the
  program will return all of the reference points with distance to the query
  point in the given range.  This is performed for an entire set of query
  points. You may specify a separate set of reference and query points, or only
  a reference set -- which is then used as both the reference and query set. 
  The given range is taken to be inclusive (that is, points with a distance
  exactly equal to the minimum and maximum of the range are included in the
  results).
  
  For example, the following will calculate the points within the range [2, 5]
  of each point in 'input.csv' and store the distances in 'distances.csv' and
  the neighbors in 'neighbors.csv':
  
  $ range_search --min=2 --max=5 --reference_file=input.csv
    --distances_file=distances.csv --neighbors_file=neighbors.csv
  
  The output files are organized such that line i corresponds to the points
  found for query point i.  Because sometimes 0 points may be found in the given
  range, lines of the output files may be empty.  The points are not ordered in
  any specific manner.
  
  Because the number of points returned for each query point may differ, the
  resultant CSV-like files may not be loadable by many programs.  However, at
  this time a better way to store this non-square result is not known.  As a
  result, any output files will be written as CSVs in this manner, regardless of
  the given extension.


  Input parameters:

   - InputModel (rsModel): File containing pre-trained range search
        model.
   - LeafSize (int): Leaf size for tree building (used for kd-trees, vp
        trees, random projection trees, UB trees, R trees, R* trees, X trees,
        Hilbert R trees, R+ trees, R++ trees, and octrees).  Default value 20.
   - Max (float64): Upper bound in range (if not specified, +inf will be
        used.  Default value 0.
   - Min (float64): Lower bound in range.  Default value 0.
   - Naive (bool): If true, O(n^2) naive mode is used for computation.
   - Query (mat.Dense): File containing query points (optional).
   - RandomBasis (bool): Before tree-building, project the data onto a
        random orthogonal basis.
   - Reference (mat.Dense): Matrix containing the reference dataset.
   - Seed (int): Random seed (if 0, std::time(NULL) is used).  Default
        value 0.
   - SingleMode (bool): If true, single-tree search is used (as opposed to
        dual-tree search).
   - TreeType (string): Type of tree to use: 'kd', 'vp', 'rp', 'max-rp',
        'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus',
        'r-plus-plus', 'oct'.  Default value 'kd'.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - distancesFile (string): File to output distances into.  Default value
        ''.
   - neighborsFile (string): File to output neighbors into.  Default value
        ''.
   - outputModel (rsModel): If specified, the range search model will be
        saved to the given file.

 */
func RangeSearch(param *RangeSearchOptionalParam) (string, string, rsModel) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Range Search")

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setRSModel("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.LeafSize != 20 {
    setParamInt("leaf_size", param.LeafSize)
    setPassed("leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Max != 0 {
    setParamDouble("max", param.Max)
    setPassed("max")
  }

  // Detect if the parameter was passed; set if so.
  if param.Min != 0 {
    setParamDouble("min", param.Min)
    setPassed("min")
  }

  // Detect if the parameter was passed; set if so.
  if param.Naive != false {
    setParamBool("naive", param.Naive)
    setPassed("naive")
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
  setPassed("distances_file")
  setPassed("neighbors_file")
  setPassed("output_model")

  // Call the mlpack program.
  C.mlpackRangeSearch()

  // Initialize result variable and get output.
  distancesFile := getParamString("distances_file")
  neighborsFile := getParamString("neighbors_file")
  var outputModel rsModel
  outputModel.getRSModel("output_model")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return distancesFile, neighborsFile, outputModel
}
