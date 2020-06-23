package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_decision_stump
#include <capi/decision_stump.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type DecisionStumpOptionalParam struct {
    BucketSize int
    InputModel *dsModel
    Labels *mat.Dense
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func DecisionStumpOptions() *DecisionStumpOptionalParam {
  return &DecisionStumpOptionalParam{
    BucketSize: 6,
    InputModel: nil,
    Labels: nil,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

type dsModel struct {
  mem unsafe.Pointer
}

func (m *dsModel) allocDSModel(identifier string) {
  m.mem = C.mlpackGetDSModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *dsModel) getDSModel(identifier string) {
  m.allocDSModel(identifier)
}

func setDSModel(identifier string, ptr *dsModel) {
  C.mlpackSetDSModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  This program implements a decision stump, which is a single-level decision
  tree.  The decision stump will split on one dimension of the input data, and
  will split into multiple buckets.  The dimension and bins are selected by
  maximizing the information gain of the split.  Optionally, the minimum number
  of training points in each bin can be specified with the "BucketSize"
  parameter.
  
  The decision stump is parameterized by a splitting dimension and a vector of
  values that denote the splitting values of each bin.
  
  This program enables several applications: a decision tree may be trained or
  loaded, and then that decision tree may be used to classify a given set of
  test points.  The decision tree may also be saved to a file for later usage.
  
  To train a decision stump, training data should be passed with the "Training"
  parameter, and their corresponding labels should be passed with the "Labels"
  option.  Optionally, if "Labels" is not specified, the labels are assumed to
  be the last dimension of the training dataset.  The "BucketSize" parameter
  controls the minimum number of training points in each decision stump bucket.
  
  For classifying a test set, a decision stump may be loaded with the
  "InputModel" parameter (useful for the situation where a stump has already
  been trained), and a test set may be specified with the "Test" parameter.  The
  predicted labels can be saved with the "Predictions" output parameter.
  
  Because decision stumps are trained in batch, retraining does not make sense
  and thus it is not possible to pass both "Training" and "InputModel"; instead,
  simply build a new decision stump with the training data.
  
  After training, a decision stump can be saved with the "OutputModel" output
  parameter.  That stump may later be re-used in subsequent calls to this
  program (or others).


  Input parameters:

   - BucketSize (int): The minimum number of training points in each
        decision stump bucket.  Default value 6.
   - InputModel (dsModel): Decision stump model to load.
   - Labels (mat.Dense): Labels for the training set. If not specified,
        the labels are assumed to be the last row of the training data.
   - Test (mat.Dense): A dataset to calculate predictions for.
   - Training (mat.Dense): The dataset to train on.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (dsModel): Output decision stump model to save.
   - predictions (mat.Dense): The output matrix that will hold the
        predicted labels for the test set.

 */
func DecisionStump(param *DecisionStumpOptionalParam) (dsModel, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Decision Stump")

  // Detect if the parameter was passed; set if so.
  if param.BucketSize != 6 {
    setParamInt("bucket_size", param.BucketSize)
    setPassed("bucket_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setDSModel("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow("labels", param.Labels)
    setPassed("labels")
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
  setPassed("predictions")

  // Call the mlpack program.
  C.mlpackDecisionStump()

  // Initialize result variable and get output.
  var outputModel dsModel
  outputModel.getDSModel("output_model")
  var predictionsPtr mlpackArma
  predictions := predictionsPtr.armaToGonumUrow("predictions")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return outputModel, predictions
}
