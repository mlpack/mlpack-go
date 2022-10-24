package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_hoeffding_tree
#include <capi/hoeffding_tree.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type HoeffdingTreeOptionalParam struct {
    BatchMode bool
    Bins int
    Confidence float64
    InfoGain bool
    InputModel *hoeffdingTreeModel
    Labels *mat.Dense
    MaxSamples int
    MinSamples int
    NumericSplitStrategy string
    ObservationsBeforeBinning int
    Passes int
    Test *matrixWithInfo
    TestLabels *mat.Dense
    Training *matrixWithInfo
    Verbose bool
}

func HoeffdingTreeOptions() *HoeffdingTreeOptionalParam {
  return &HoeffdingTreeOptionalParam{
    BatchMode: false,
    Bins: 10,
    Confidence: 0.95,
    InfoGain: false,
    InputModel: nil,
    Labels: nil,
    MaxSamples: 5000,
    MinSamples: 100,
    NumericSplitStrategy: "binary",
    ObservationsBeforeBinning: 100,
    Passes: 1,
    Test: nil,
    TestLabels: nil,
    Training: nil,
    Verbose: false,
  }
}

/*
  This program implements Hoeffding trees, a form of streaming decision tree
  suited best for large (or streaming) datasets.  This program supports both
  categorical and numeric data.  Given an input dataset, this program is able to
  train the tree with numerous training options, and save the model to a file. 
  The program is also able to use a trained model or a model from file in order
  to predict classes for a given test set.
  
  The training file and associated labels are specified with the "Training" and
  "Labels" parameters, respectively. Optionally, if "Labels" is not specified,
  the labels are assumed to be the last dimension of the training dataset.
  
  The training may be performed in batch mode (like a typical decision tree
  algorithm) by specifying the "BatchMode" option, but this may not be the best
  option for large datasets.
  
  When a model is trained, it may be saved via the "OutputModel" output
  parameter.  A model may be loaded from file for further training or testing
  with the "InputModel" parameter.
  
  Test data may be specified with the "Test" parameter, and if performance
  statistics are desired for that test set, labels may be specified with the
  "TestLabels" parameter.  Predictions for each test point may be saved with the
  "Predictions" output parameter, and class probabilities for each prediction
  may be saved with the "Probabilities" output parameter.

  For example, to train a Hoeffding tree with confidence 0.99 with data dataset,
  saving the trained tree to tree, the following command may be used:
  
  // Initialize optional parameters for HoeffdingTree().
  param := mlpack.HoeffdingTreeOptions()
  param.Training = dataset
  param.Confidence = 0.99
  
  tree, _, _ := mlpack.HoeffdingTree(param)
  
  Then, this tree may be used to make predictions on the test set test_set,
  saving the predictions into predictions and the class probabilities into
  class_probs with the following command: 
  
  // Initialize optional parameters for HoeffdingTree().
  param := mlpack.HoeffdingTreeOptions()
  param.InputModel = &tree
  param.Test = test_set
  
  _, predictions, class_probs := mlpack.HoeffdingTree(param)

  Input parameters:

   - BatchMode (bool): If true, samples will be considered in batch
        instead of as a stream.  This generally results in better trees but at
        the cost of memory usage and runtime.
   - Bins (int): If the 'domingos' split strategy is used, this specifies
        the number of bins for each numeric split.  Default value 10.
   - Confidence (float64): Confidence before splitting (between 0 and 1). 
        Default value 0.95.
   - InfoGain (bool): If set, information gain is used instead of Gini
        impurity for calculating Hoeffding bounds.
   - InputModel (hoeffdingTreeModel): Input trained Hoeffding tree model.
   - Labels (mat.Dense): Labels for training dataset.
   - MaxSamples (int): Maximum number of samples before splitting. 
        Default value 5000.
   - MinSamples (int): Minimum number of samples before splitting. 
        Default value 100.
   - NumericSplitStrategy (string): The splitting strategy to use for
        numeric features: 'domingos' or 'binary'.  Default value 'binary'.
   - ObservationsBeforeBinning (int): If the 'domingos' split strategy is
        used, this specifies the number of samples observed before binning is
        performed.  Default value 100.
   - Passes (int): Number of passes to take over the dataset.  Default
        value 1.
   - Test (matrixWithInfo): Testing dataset (may be categorical).
   - TestLabels (mat.Dense): Labels of test data.
   - Training (matrixWithInfo): Training dataset (may be categorical).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (hoeffdingTreeModel): Output for trained Hoeffding tree
        model.
   - predictions (mat.Dense): Matrix to output label predictions for test
        data into.
   - probabilities (mat.Dense): In addition to predicting labels, provide
        rediction probabilities in this matrix.

 */
func HoeffdingTree(param *HoeffdingTreeOptionalParam) (hoeffdingTreeModel, *mat.Dense, *mat.Dense) {
  params := getParams("hoeffding_tree")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.BatchMode != false {
    setParamBool(params, "batch_mode", param.BatchMode)
    setPassed(params, "batch_mode")
  }

  // Detect if the parameter was passed; set if so.
  if param.Bins != 10 {
    setParamInt(params, "bins", param.Bins)
    setPassed(params, "bins")
  }

  // Detect if the parameter was passed; set if so.
  if param.Confidence != 0.95 {
    setParamDouble(params, "confidence", param.Confidence)
    setPassed(params, "confidence")
  }

  // Detect if the parameter was passed; set if so.
  if param.InfoGain != false {
    setParamBool(params, "info_gain", param.InfoGain)
    setPassed(params, "info_gain")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setHoeffdingTreeModel(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow(params, "labels", param.Labels)
    setPassed(params, "labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxSamples != 5000 {
    setParamInt(params, "max_samples", param.MaxSamples)
    setPassed(params, "max_samples")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinSamples != 100 {
    setParamInt(params, "min_samples", param.MinSamples)
    setPassed(params, "min_samples")
  }

  // Detect if the parameter was passed; set if so.
  if param.NumericSplitStrategy != "binary" {
    setParamString(params, "numeric_split_strategy", param.NumericSplitStrategy)
    setPassed(params, "numeric_split_strategy")
  }

  // Detect if the parameter was passed; set if so.
  if param.ObservationsBeforeBinning != 100 {
    setParamInt(params, "observations_before_binning", param.ObservationsBeforeBinning)
    setPassed(params, "observations_before_binning")
  }

  // Detect if the parameter was passed; set if so.
  if param.Passes != 1 {
    setParamInt(params, "passes", param.Passes)
    setPassed(params, "passes")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMatWithInfo(params, "test", param.Test)
    setPassed(params, "test")
  }

  // Detect if the parameter was passed; set if so.
  if param.TestLabels != nil {
    gonumToArmaUrow(params, "test_labels", param.TestLabels)
    setPassed(params, "test_labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    gonumToArmaMatWithInfo(params, "training", param.Training)
    setPassed(params, "training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "output_model")
  setPassed(params, "predictions")
  setPassed(params, "probabilities")

  // Call the mlpack program.
  C.mlpackHoeffdingTree(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputModel hoeffdingTreeModel
  outputModel.getHoeffdingTreeModel(params, "output_model")
  var predictionsPtr mlpackArma
  predictions := predictionsPtr.armaToGonumUrow(params, "predictions")
  var probabilitiesPtr mlpackArma
  probabilities := probabilitiesPtr.armaToGonumMat(params, "probabilities")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return outputModel, predictions, probabilities
}
