package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_decision_tree
#include <capi/decision_tree.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type DecisionTreeOptionalParam struct {
    InputModel *decisionTreeModel
    Labels *mat.Dense
    MaximumDepth int
    MinimumGainSplit float64
    MinimumLeafSize int
    PrintTrainingAccuracy bool
    PrintTrainingError bool
    Test *matrixWithInfo
    TestLabels *mat.Dense
    Training *matrixWithInfo
    Verbose bool
    Weights *mat.Dense
}

func DecisionTreeOptions() *DecisionTreeOptionalParam {
  return &DecisionTreeOptionalParam{
    InputModel: nil,
    Labels: nil,
    MaximumDepth: 0,
    MinimumGainSplit: 1e-07,
    MinimumLeafSize: 20,
    PrintTrainingAccuracy: false,
    PrintTrainingError: false,
    Test: nil,
    TestLabels: nil,
    Training: nil,
    Verbose: false,
    Weights: nil,
  }
}

type decisionTreeModel struct {
  mem unsafe.Pointer
}

func (m *decisionTreeModel) allocDecisionTreeModel(identifier string) {
  m.mem = C.mlpackGetDecisionTreeModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *decisionTreeModel) getDecisionTreeModel(identifier string) {
  m.allocDecisionTreeModel(identifier)
}

func setDecisionTreeModel(identifier string, ptr *decisionTreeModel) {
  C.mlpackSetDecisionTreeModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  Train and evaluate using a decision tree.  Given a dataset containing numeric
  or categorical features, and associated labels for each point in the dataset,
  this program can train a decision tree on that data.
  
  The training set and associated labels are specified with the "Training" and
  "Labels" parameters, respectively.  The labels should be in the range [0,
  num_classes - 1]. Optionally, if "Labels" is not specified, the labels are
  assumed to be the last dimension of the training dataset.
  
  When a model is trained, the "OutputModel" output parameter may be used to
  save the trained model.  A model may be loaded for predictions with the
  "InputModel" parameter.  The "InputModel" parameter may not be specified when
  the "Training" parameter is specified.  The "MinimumLeafSize" parameter
  specifies the minimum number of training points that must fall into each leaf
  for it to be split.  The "MinimumGainSplit" parameter specifies the minimum
  gain that is needed for the node to split.  The "MaximumDepth" parameter
  specifies the maximum depth of the tree.  If "PrintTrainingError" is
  specified, the training error will be printed.
  
  Test data may be specified with the "Test" parameter, and if performance
  numbers are desired for that test set, labels may be specified with the
  "TestLabels" parameter.  Predictions for each test point may be saved via the
  "Predictions" output parameter.  Class probabilities for each prediction may
  be saved with the "Probabilities" output parameter.
  
  For example, to train a decision tree with a minimum leaf size of 20 on the
  dataset contained in data with labels labels, saving the output model to tree
  and printing the training error, one could call
  
  // Initialize optional parameters for DecisionTree().
  param := mlpack.DecisionTreeOptions()
  param.Training = data
  param.Labels = labels
  param.MinimumLeafSize = 20
  param.MinimumGainSplit = 0.001
  param.PrintTrainingAccuracy = true
  
  tree, _, _ := mlpack.DecisionTree(param)
  
  Then, to use that model to classify points in test_set and print the test
  error given the labels test_labels using that model, while saving the
  predictions for each point to predictions, one could call 
  
  // Initialize optional parameters for DecisionTree().
  param := mlpack.DecisionTreeOptions()
  param.InputModel = &tree
  param.Test = test_set
  param.TestLabels = test_labels
  
  _, predictions, _ := mlpack.DecisionTree(param)


  Input parameters:

   - InputModel (decisionTreeModel): Pre-trained decision tree, to be used
        with test points.
   - Labels (mat.Dense): Training labels.
   - MaximumDepth (int): Maximum depth of the tree (0 means no limit). 
        Default value 0.
   - MinimumGainSplit (float64): Minimum gain for node splitting.  Default
        value 1e-07.
   - MinimumLeafSize (int): Minimum number of points in a leaf.  Default
        value 20.
   - PrintTrainingAccuracy (bool): Print the training accuracy.
   - PrintTrainingError (bool): Print the training error (deprecated; will
        be removed in mlpack 4.0.0).
   - Test (matrixWithInfo): Testing dataset (may be categorical).
   - TestLabels (mat.Dense): Test point labels, if accuracy calculation is
        desired.
   - Training (matrixWithInfo): Training dataset (may be categorical).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.
   - Weights (mat.Dense): The weight of labels

  Output parameters:

   - outputModel (decisionTreeModel): Output for trained decision tree.
   - predictions (mat.Dense): Class predictions for each test point.
   - probabilities (mat.Dense): Class probabilities for each test point.

 */
func DecisionTree(param *DecisionTreeOptionalParam) (decisionTreeModel, *mat.Dense, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Decision tree")

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setDecisionTreeModel("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow("labels", param.Labels)
    setPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaximumDepth != 0 {
    setParamInt("maximum_depth", param.MaximumDepth)
    setPassed("maximum_depth")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinimumGainSplit != 1e-07 {
    setParamDouble("minimum_gain_split", param.MinimumGainSplit)
    setPassed("minimum_gain_split")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinimumLeafSize != 20 {
    setParamInt("minimum_leaf_size", param.MinimumLeafSize)
    setPassed("minimum_leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.PrintTrainingAccuracy != false {
    setParamBool("print_training_accuracy", param.PrintTrainingAccuracy)
    setPassed("print_training_accuracy")
  }

  // Detect if the parameter was passed; set if so.
  if param.PrintTrainingError != false {
    setParamBool("print_training_error", param.PrintTrainingError)
    setPassed("print_training_error")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMatWithInfo("test", param.Test)
    setPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.TestLabels != nil {
    gonumToArmaUrow("test_labels", param.TestLabels)
    setPassed("test_labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    gonumToArmaMatWithInfo("training", param.Training)
    setPassed("training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Detect if the parameter was passed; set if so.
  if param.Weights != nil {
    gonumToArmaMat("weights", param.Weights)
    setPassed("weights")
  }

  // Mark all output options as passed.
  setPassed("output_model")
  setPassed("predictions")
  setPassed("probabilities")

  // Call the mlpack program.
  C.mlpackDecisionTree()

  // Initialize result variable and get output.
  var outputModel decisionTreeModel
  outputModel.getDecisionTreeModel("output_model")
  var predictionsPtr mlpackArma
  predictions := predictionsPtr.armaToGonumUrow("predictions")
  var probabilitiesPtr mlpackArma
  probabilities := probabilitiesPtr.armaToGonumMat("probabilities")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return outputModel, predictions, probabilities
}
