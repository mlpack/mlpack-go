package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_det
#include <capi/det.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type DetOptionalParam struct {
    Folds int
    InputModel *dTree
    MaxLeafSize int
    MinLeafSize int
    PathFormat string
    SkipPruning bool
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func DetOptions() *DetOptionalParam {
  return &DetOptionalParam{
    Folds: 10,
    InputModel: nil,
    MaxLeafSize: 10,
    MinLeafSize: 5,
    PathFormat: "lr",
    SkipPruning: false,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type dTree struct {
  mem unsafe.Pointer
}

func (m *dTree) allocDTree(identifier string) {
  m.mem = C.mlpackGetDTreePtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *dTree) getDTree(identifier string) {
  m.allocDTree(identifier)
}

func setDTree(identifier string, ptr *dTree) {
  C.mlpackSetDTreePtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program performs a number of functions related to Density Estimation
  Trees.  The optimal Density Estimation Tree (DET) can be trained on a set of
  data (specified by "Training") using cross-validation (with number of folds
  specified with the "Folds" parameter).  This trained density estimation tree
  may then be saved with the "OutputModel" output parameter.
  
  The variable importances (that is, the feature importance values for each
  dimension) may be saved with the "Vi" output parameter, and the density
  estimates for each training point may be saved with the "TrainingSetEstimates"
  output parameter.
  
  Enabling path printing for each node outputs the path from the root node to a
  leaf for each entry in the test set, or training set (if a test set is not
  provided).  Strings like 'LRLRLR' (indicating that traversal went to the left
  child, then the right child, then the left child, and so forth) will be
  output. If 'lr-id' or 'id-lr' are given as the "PathFormat" parameter, then
  the ID (tag) of every node along the path will be printed after or before the
  L or R character indicating the direction of traversal, respectively.
  
  This program also can provide density estimates for a set of test points,
  specified in the "Test" parameter.  The density estimation tree used for this
  task will be the tree that was trained on the given training points, or a tree
  given as the parameter "InputModel".  The density estimates for the test
  points may be saved using the "TestSetEstimates" output parameter.


  Input parameters:

   - Folds (int): The number of folds of cross-validation to perform for
        the estimation (0 is LOOCV)  Default value 10.
   - InputModel (dTree): Trained density estimation tree to load.
   - MaxLeafSize (int): The maximum size of a leaf in the unpruned, fully
        grown DET.  Default value 10.
   - MinLeafSize (int): The minimum size of a leaf in the unpruned, fully
        grown DET.  Default value 5.
   - PathFormat (string): The format of path printing: 'lr', 'id-lr', or
        'lr-id'.  Default value 'lr'.
   - SkipPruning (bool): Whether to bypass the pruning process and output
        the unpruned tree only.
   - Test (mat.Dense): A set of test points to estimate the density of.
   - Training (mat.Dense): The data set on which to build a density
        estimation tree.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (dTree): Output to save trained density estimation tree
        to.
   - tagCountersFile (string): The file to output the number of points
        that went to each leaf.  Default value ''.
   - tagFile (string): The file to output the tags (and possibly paths)
        for each sample in the test set.  Default value ''.
   - testSetEstimates (mat.Dense): The output estimates on the test set
        from the final optimally pruned tree.
   - trainingSetEstimates (mat.Dense): The output density estimates on the
        training set from the final optimally pruned tree.
   - vi (mat.Dense): The output variable importance values for each
        feature.

 */
func Det(param *DetOptionalParam) (dTree, string, string, *mat.Dense, *mat.Dense, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Density Estimation With Density Estimation Trees")

  // Detect if the parameter was passed; set if so.
  if param.Folds != 10 {
    setParamInt("folds", param.Folds)
    setPassed("folds")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setDTree("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxLeafSize != 10 {
    setParamInt("max_leaf_size", param.MaxLeafSize)
    setPassed("max_leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinLeafSize != 5 {
    setParamInt("min_leaf_size", param.MinLeafSize)
    setPassed("min_leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.PathFormat != "lr" {
    setParamString("path_format", param.PathFormat)
    setPassed("path_format")
  }

  // Detect if the parameter was passed; set if so.
  if param.SkipPruning != false {
    setParamBool("skip_pruning", param.SkipPruning)
    setPassed("skip_pruning")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat("test", param.Test)
    setPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    gonumToArmaMat("training", param.Training)
    setPassed("training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("output_model")
  setPassed("tag_counters_file")
  setPassed("tag_file")
  setPassed("test_set_estimates")
  setPassed("training_set_estimates")
  setPassed("vi")

  // Call the mlpack program.
  C.mlpackDet()

  // Initialize result variable and get output.
  var outputModel dTree
  outputModel.getDTree("output_model")
  tagCountersFile := getParamString("tag_counters_file")
  tagFile := getParamString("tag_file")
  var testSetEstimatesPtr mlpackArma
  testSetEstimates := testSetEstimatesPtr.armaToGonumMat("test_set_estimates")
  var trainingSetEstimatesPtr mlpackArma
  trainingSetEstimates := trainingSetEstimatesPtr.armaToGonumMat("training_set_estimates")
  var viPtr mlpackArma
  vi := viPtr.armaToGonumMat("vi")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return outputModel, tagCountersFile, tagFile, testSetEstimates, trainingSetEstimates, vi
}
