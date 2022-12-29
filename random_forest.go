package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_random_forest
#include <capi/random_forest.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type RandomForestOptionalParam struct {
    InputModel *randomForestModel
    Labels *mat.Dense
    MaximumDepth int
    MinimumGainSplit float64
    MinimumLeafSize int
    NumTrees int
    PrintTrainingAccuracy bool
    Seed int
    SubspaceDim int
    Test *mat.Dense
    TestLabels *mat.Dense
    Training *mat.Dense
    Verbose bool
    WarmStart bool
}

func RandomForestOptions() *RandomForestOptionalParam {
  return &RandomForestOptionalParam{
    InputModel: nil,
    Labels: nil,
    MaximumDepth: 0,
    MinimumGainSplit: 0,
    MinimumLeafSize: 1,
    NumTrees: 10,
    PrintTrainingAccuracy: false,
    Seed: 0,
    SubspaceDim: 0,
    Test: nil,
    TestLabels: nil,
    Training: nil,
    Verbose: false,
    WarmStart: false,
  }
}

/*
  This program is an implementation of the standard random forest classification
  algorithm by Leo Breiman.  A random forest can be trained and saved for later
  use, or a random forest may be loaded and predictions or class probabilities
  for points may be generated.
  
  The training set and associated labels are specified with the "Training" and
  "Labels" parameters, respectively.  The labels should be in the range [0,
  num_classes - 1]. Optionally, if "Labels" is not specified, the labels are
  assumed to be the last dimension of the training dataset.
  
  When a model is trained, the "OutputModel" output parameter may be used to
  save the trained model.  A model may be loaded for predictions with the
  "InputModel"parameter. The "InputModel" parameter may not be specified when
  the "Training" parameter is specified.  The "MinimumLeafSize" parameter
  specifies the minimum number of training points that must fall into each leaf
  for it to be split.  The "NumTrees" controls the number of trees in the random
  forest.  The "MinimumGainSplit" parameter controls the minimum required gain
  for a decision tree node to split.  Larger values will force higher-confidence
  splits.  The "MaximumDepth" parameter specifies the maximum depth of the tree.
   The "SubspaceDim" parameter is used to control the number of random
  dimensions chosen for an individual node's split.  If "PrintTrainingAccuracy"
  is specified, the calculated accuracy on the training set will be printed.
  
  Test data may be specified with the "Test" parameter, and if performance
  measures are desired for that test set, labels for the test points may be
  specified with the "TestLabels" parameter.  Predictions for each test point
  may be saved via the "Predictions"output parameter.  Class probabilities for
  each prediction may be saved with the "Probabilities" output parameter.

  For example, to train a random forest with a minimum leaf size of 20 using 10
  trees on the dataset contained in datawith labels labels, saving the output
  random forest to rf_model and printing the training error, one could call
  
  // Initialize optional parameters for RandomForest().
  param := mlpack.RandomForestOptions()
  param.Training = data
  param.Labels = labels
  param.MinimumLeafSize = 20
  param.NumTrees = 10
  param.PrintTrainingAccuracy = true
  
  rf_model, _, _ := mlpack.RandomForest(param)
  
  Then, to use that model to classify points in test_set and print the test
  error given the labels test_labels using that model, while saving the
  predictions for each point to predictions, one could call 
  
  // Initialize optional parameters for RandomForest().
  param := mlpack.RandomForestOptions()
  param.InputModel = &rf_model
  param.Test = test_set
  param.TestLabels = test_labels
  
  _, predictions, _ := mlpack.RandomForest(param)

  Input parameters:

   - InputModel (randomForestModel): Pre-trained random forest to use for
        classification.
   - Labels (mat.Dense): Labels for training dataset.
   - MaximumDepth (int): Maximum depth of the tree (0 means no limit). 
        Default value 0.
   - MinimumGainSplit (float64): Minimum gain needed to make a split when
        building a tree.  Default value 0.
   - MinimumLeafSize (int): Minimum number of points in each leaf node. 
        Default value 1.
   - NumTrees (int): Number of trees in the random forest.  Default value
        10.
   - PrintTrainingAccuracy (bool): If set, then the accuracy of the model
        on the training set will be predicted (verbose must also be specified).
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - SubspaceDim (int): Dimensionality of random subspace to use for each
        split.  '0' will autoselect the square root of data dimensionality. 
        Default value 0.
   - Test (mat.Dense): Test dataset to produce predictions for.
   - TestLabels (mat.Dense): Test dataset labels, if accuracy calculation
        is desired.
   - Training (mat.Dense): Training dataset.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.
   - WarmStart (bool): If true and passed along with `training` and
        `input_model` then trains more trees on top of existing model.

  Output parameters:

   - outputModel (randomForestModel): Model to save trained random forest
        to.
   - predictions (mat.Dense): Predicted classes for each point in the test
        set.
   - probabilities (mat.Dense): Predicted class probabilities for each
        point in the test set.

 */
func RandomForest(param *RandomForestOptionalParam) (randomForestModel, *mat.Dense, *mat.Dense) {
  params := getParams("random_forest")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setRandomForestModel(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow(params, "labels", param.Labels)
    setPassed(params, "labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaximumDepth != 0 {
    setParamInt(params, "maximum_depth", param.MaximumDepth)
    setPassed(params, "maximum_depth")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinimumGainSplit != 0 {
    setParamDouble(params, "minimum_gain_split", param.MinimumGainSplit)
    setPassed(params, "minimum_gain_split")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinimumLeafSize != 1 {
    setParamInt(params, "minimum_leaf_size", param.MinimumLeafSize)
    setPassed(params, "minimum_leaf_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.NumTrees != 10 {
    setParamInt(params, "num_trees", param.NumTrees)
    setPassed(params, "num_trees")
  }

  // Detect if the parameter was passed; set if so.
  if param.PrintTrainingAccuracy != false {
    setParamBool(params, "print_training_accuracy", param.PrintTrainingAccuracy)
    setPassed(params, "print_training_accuracy")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt(params, "seed", param.Seed)
    setPassed(params, "seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.SubspaceDim != 0 {
    setParamInt(params, "subspace_dim", param.SubspaceDim)
    setPassed(params, "subspace_dim")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat(params, "test", param.Test, false)
    setPassed(params, "test")
  }

  // Detect if the parameter was passed; set if so.
  if param.TestLabels != nil {
    gonumToArmaUrow(params, "test_labels", param.TestLabels)
    setPassed(params, "test_labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Training != nil {
    gonumToArmaMat(params, "training", param.Training, false)
    setPassed(params, "training")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Detect if the parameter was passed; set if so.
  if param.WarmStart != false {
    setParamBool(params, "warm_start", param.WarmStart)
    setPassed(params, "warm_start")
  }

  // Mark all output options as passed.
  setPassed(params, "output_model")
  setPassed(params, "predictions")
  setPassed(params, "probabilities")

  // Call the mlpack program.
  C.mlpackRandomForest(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputModel randomForestModel
  outputModel.getRandomForestModel(params, "output_model")
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
