package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_linear_svm
#include <capi/linear_svm.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

type LinearSvmOptionalParam struct {
    Delta float64
    Epochs int
    InputModel *linearsvmModel
    Labels *mat.Dense
    Lambda float64
    MaxIterations int
    NoIntercept bool
    NumClasses int
    Optimizer string
    Seed int
    Shuffle bool
    StepSize float64
    Test *mat.Dense
    TestLabels *mat.Dense
    Tolerance float64
    Training *mat.Dense
    Verbose bool
}

func LinearSvmOptions() *LinearSvmOptionalParam {
  return &LinearSvmOptionalParam{
    Delta: 1,
    Epochs: 50,
    InputModel: nil,
    Labels: nil,
    Lambda: 0.0001,
    MaxIterations: 10000,
    NoIntercept: false,
    NumClasses: 0,
    Optimizer: "lbfgs",
    Seed: 0,
    Shuffle: false,
    StepSize: 0.01,
    Test: nil,
    TestLabels: nil,
    Tolerance: 1e-10,
    Training: nil,
    Verbose: false,
  }
}

type linearsvmModel struct {
  mem unsafe.Pointer
}

func (m *linearsvmModel) allocLinearSVMModel(identifier string) {
  m.mem = C.mlpackGetLinearSVMModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *linearsvmModel) getLinearSVMModel(identifier string) {
  m.allocLinearSVMModel(identifier)
}

func setLinearSVMModel(identifier string, ptr *linearsvmModel) {
  C.mlpackSetLinearSVMModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  An implementation of linear SVMs that uses either L-BFGS or parallel SGD
  (stochastic gradient descent) to train the model.
  
  This program allows loading a linear SVM model (via the "InputModel"
  parameter) or training a linear SVM model given training data (specified with
  the "Training" parameter), or both those things at once.  In addition, this
  program allows classification on a test dataset (specified with the "Test"
  parameter) and the classification results may be saved with the "Predictions"
  output parameter. The trained linear SVM model may be saved using the
  "OutputModel" output parameter.
  
  The training data, if specified, may have class labels as its last dimension. 
  Alternately, the "Labels" parameter may be used to specify a separate vector
  of labels.
  
  When a model is being trained, there are many options.  L2 regularization (to
  prevent overfitting) can be specified with the "Lambda" option, and the number
  of classes can be manually specified with the "NumClasses"and if an intercept
  term is not desired in the model, the "NoIntercept" parameter can be
  specified.Margin of difference between correct class and other classes can be
  specified with the "Delta" option.The optimizer used to train the model can be
  specified with the "Optimizer" parameter.  Available options are 'psgd'
  (parallel stochastic gradient descent) and 'lbfgs' (the L-BFGS optimizer). 
  There are also various parameters for the optimizer; the "MaxIterations"
  parameter specifies the maximum number of allowed iterations, and the
  "Tolerance" parameter specifies the tolerance for convergence.  For the
  parallel SGD optimizer, the "StepSize" parameter controls the step size taken
  at each iteration by the optimizer and the maximum number of epochs (specified
  with "Epochs"). If the objective function for your data is oscillating between
  Inf and 0, the step size is probably too large.  There are more parameters for
  the optimizers, but the C++ interface must be used to access these.
  
  Optionally, the model can be used to predict the labels for another matrix of
  data points, if "Test" is specified.  The "Test" parameter can be specified
  without the "Training" parameter, so long as an existing linear SVM model is
  given with the "InputModel" parameter.  The output predictions from the linear
  SVM model may be saved with the "Predictions" parameter.
  
  As an example, to train a LinaerSVM on the data 'data' with labels 'labels'
  with L2 regularization of 0.1, saving the model to 'lsvm_model', the following
  command may be used:
  
  // Initialize optional parameters for LinearSvm().
  param := mlpack.LinearSvmOptions()
  param.Training = data
  param.Labels = labels
  param.Lambda = 0.1
  param.Delta = 1
  param.NumClasses = 0
  
  lsvm_model, _, _ := mlpack.LinearSvm(param)
  
  Then, to use that model to predict classes for the dataset 'test', storing the
  output predictions in 'predictions', the following command may be used: 
  
  // Initialize optional parameters for LinearSvm().
  param := mlpack.LinearSvmOptions()
  param.InputModel = &lsvm_model
  param.Test = test
  
  _, predictions, _ := mlpack.LinearSvm(param)


  Input parameters:

   - Delta (float64): Margin of difference between correct class and other
        classes.  Default value 1.
   - Epochs (int): Maximum number of full epochs over dataset for psgd 
        Default value 50.
   - InputModel (linearsvmModel): Existing model (parameters).
   - Labels (mat.Dense): A matrix containing labels (0 or 1) for the
        points in the training set (y).
   - Lambda (float64): L2-regularization parameter for training.  Default
        value 0.0001.
   - MaxIterations (int): Maximum iterations for optimizer (0 indicates no
        limit).  Default value 10000.
   - NoIntercept (bool): Do not add the intercept term to the model.
   - NumClasses (int): Number of classes for classification; if
        unspecified (or 0), the number of classes found in the labels will be
        used.  Default value 0.
   - Optimizer (string): Optimizer to use for training ('lbfgs' or
        'psgd').  Default value 'lbfgs'.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - Shuffle (bool): Don't shuffle the order in which data points are
        visited for parallel SGD.
   - StepSize (float64): Step size for parallel SGD optimizer.  Default
        value 0.01.
   - Test (mat.Dense): Matrix containing test dataset.
   - TestLabels (mat.Dense): Matrix containing test labels.
   - Tolerance (float64): Convergence tolerance for optimizer.  Default
        value 1e-10.
   - Training (mat.Dense): A matrix containing the training set (the
        matrix of predictors, X).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - outputModel (linearsvmModel): Output for trained linear svm model.
   - predictions (mat.Dense): If test data is specified, this matrix is
        where the predictions for the test set will be saved.
   - probabilities (mat.Dense): If test data is specified, this matrix is
        where the class probabilities for the test set will be saved.

 */
func LinearSvm(param *LinearSvmOptionalParam) (linearsvmModel, *mat.Dense, *mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Linear SVM is an L2-regularized support vector machine.")

  // Detect if the parameter was passed; set if so.
  if param.Delta != 1 {
    setParamDouble("delta", param.Delta)
    setPassed("delta")
  }

  // Detect if the parameter was passed; set if so.
  if param.Epochs != 50 {
    setParamInt("epochs", param.Epochs)
    setPassed("epochs")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setLinearSVMModel("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Labels != nil {
    gonumToArmaUrow("labels", param.Labels)
    setPassed("labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda != 0.0001 {
    setParamDouble("lambda", param.Lambda)
    setPassed("lambda")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 10000 {
    setParamInt("max_iterations", param.MaxIterations)
    setPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.NoIntercept != false {
    setParamBool("no_intercept", param.NoIntercept)
    setPassed("no_intercept")
  }

  // Detect if the parameter was passed; set if so.
  if param.NumClasses != 0 {
    setParamInt("num_classes", param.NumClasses)
    setPassed("num_classes")
  }

  // Detect if the parameter was passed; set if so.
  if param.Optimizer != "lbfgs" {
    setParamString("optimizer", param.Optimizer)
    setPassed("optimizer")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Shuffle != false {
    setParamBool("shuffle", param.Shuffle)
    setPassed("shuffle")
  }

  // Detect if the parameter was passed; set if so.
  if param.StepSize != 0.01 {
    setParamDouble("step_size", param.StepSize)
    setPassed("step_size")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat("test", param.Test)
    setPassed("test")
  }

  // Detect if the parameter was passed; set if so.
  if param.TestLabels != nil {
    gonumToArmaUrow("test_labels", param.TestLabels)
    setPassed("test_labels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Tolerance != 1e-10 {
    setParamDouble("tolerance", param.Tolerance)
    setPassed("tolerance")
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
  setPassed("probabilities")

  // Call the mlpack program.
  C.mlpackLinearSvm()

  // Initialize result variable and get output.
  var outputModel linearsvmModel
  outputModel.getLinearSVMModel("output_model")
  var predictionsPtr mlpackArma
  predictions := predictionsPtr.armaToGonumUrow("predictions")
  var probabilitiesPtr mlpackArma
  probabilities := probabilitiesPtr.armaToGonumMat("probabilities")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return outputModel, predictions, probabilities
}
