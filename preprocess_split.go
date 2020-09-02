package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_preprocess_split
#include <capi/preprocess_split.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type PreprocessSplitOptionalParam struct {
    InputLabels *mat.Dense
    NoShuffle bool
    Seed int
    TestRatio float64
    Verbose bool
}

func PreprocessSplitOptions() *PreprocessSplitOptionalParam {
  return &PreprocessSplitOptionalParam{
    InputLabels: nil,
    NoShuffle: false,
    Seed: 0,
    TestRatio: 0.2,
    Verbose: false,
  }
}

/*
  This utility takes a dataset and optionally labels and splits them into a
  training set and a test set. Before the split, the points in the dataset are
  randomly reordered. The percentage of the dataset to be used as the test set
  can be specified with the "TestRatio" parameter; the default is 0.2 (20%).
  
  The output training and test matrices may be saved with the "Training" and
  "Test" output parameters.
  
  Optionally, labels can be also be split along with the data by specifying the
  "InputLabels" parameter.  Splitting labels works the same way as splitting the
  data. The output training and test labels may be saved with the
  "TrainingLabels" and "TestLabels" output parameters, respectively.

  So, a simple example where we want to split the dataset X into X_train and
  X_test with 60% of the data in the training set and 40% of the dataset in the
  test set, we could run 
  
  // Initialize optional parameters for PreprocessSplit().
  param := mlpack.PreprocessSplitOptions()
  param.TestRatio = 0.4
  
  X_test, _, X_train, _ := mlpack.PreprocessSplit(X, param)
  
  Also by default the dataset is shuffled and split; you can provide the
  "NoShuffle" option to avoid shuffling the data; an example to avoid shuffling
  of data is:
  
  // Initialize optional parameters for PreprocessSplit().
  param := mlpack.PreprocessSplitOptions()
  param.TestRatio = 0.4
  param.NoShuffle = true
  
  X_test, _, X_train, _ := mlpack.PreprocessSplit(X, param)
  
  If we had a dataset X and associated labels y, and we wanted to split these
  into X_train, y_train, X_test, and y_test, with 30% of the data in the test
  set, we could run
  
  // Initialize optional parameters for PreprocessSplit().
  param := mlpack.PreprocessSplitOptions()
  param.InputLabels = y
  param.TestRatio = 0.3
  
  X_test, y_test, X_train, y_train := mlpack.PreprocessSplit(X, param)

  Input parameters:

   - input (mat.Dense): Matrix containing data.
   - InputLabels (mat.Dense): Matrix containing labels.
   - NoShuffle (bool): Avoid shuffling and splitting the data.
   - Seed (int): Random seed (0 for std::time(NULL)).  Default value 0.
   - TestRatio (float64): Ratio of test set; if not set,the ratio defaults
        to 0.2  Default value 0.2.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - test (mat.Dense): Matrix to save test data to.
   - testLabels (mat.Dense): Matrix to save test labels to.
   - training (mat.Dense): Matrix to save training data to.
   - trainingLabels (mat.Dense): Matrix to save train labels to.

 */
func PreprocessSplit(input *mat.Dense, param *PreprocessSplitOptionalParam) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Split Data")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.InputLabels != nil {
    gonumToArmaUmat("input_labels", param.InputLabels)
    setPassed("input_labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.NoShuffle != false {
    setParamBool("no_shuffle", param.NoShuffle)
    setPassed("no_shuffle")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.TestRatio != 0.2 {
    setParamDouble("test_ratio", param.TestRatio)
    setPassed("test_ratio")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("test")
  setPassed("test_labels")
  setPassed("training")
  setPassed("training_labels")

  // Call the mlpack program.
  C.mlpackPreprocessSplit()

  // Initialize result variable and get output.
  var testPtr mlpackArma
  test := testPtr.armaToGonumMat("test")
  var testLabelsPtr mlpackArma
  testLabels := testLabelsPtr.armaToGonumUmat("test_labels")
  var trainingPtr mlpackArma
  training := trainingPtr.armaToGonumMat("training")
  var trainingLabelsPtr mlpackArma
  trainingLabels := trainingLabelsPtr.armaToGonumUmat("training_labels")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return test, testLabels, training, trainingLabels
}
