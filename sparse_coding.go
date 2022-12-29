package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_sparse_coding
#include <capi/sparse_coding.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type SparseCodingOptionalParam struct {
    Atoms int
    InitialDictionary *mat.Dense
    InputModel *sparseCoding
    Lambda1 float64
    Lambda2 float64
    MaxIterations int
    NewtonTolerance float64
    Normalize bool
    ObjectiveTolerance float64
    Seed int
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func SparseCodingOptions() *SparseCodingOptionalParam {
  return &SparseCodingOptionalParam{
    Atoms: 15,
    InitialDictionary: nil,
    InputModel: nil,
    Lambda1: 0,
    Lambda2: 0,
    MaxIterations: 0,
    NewtonTolerance: 1e-06,
    Normalize: false,
    ObjectiveTolerance: 0.01,
    Seed: 0,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

/*
  An implementation of Sparse Coding with Dictionary Learning, which achieves
  sparsity via an l1-norm regularizer on the codes (LASSO) or an (l1+l2)-norm
  regularizer on the codes (the Elastic Net).  Given a dense data matrix X with
  d dimensions and n points, sparse coding seeks to find a dense dictionary
  matrix D with k atoms in d dimensions, and a sparse coding matrix Z with n
  points in k dimensions.
  
  The original data matrix X can then be reconstructed as Z * D.  Therefore,
  this program finds a representation of each point in X as a sparse linear
  combination of atoms in the dictionary D.
  
  The sparse coding is found with an algorithm which alternates between a
  dictionary step, which updates the dictionary D, and a sparse coding step,
  which updates the sparse coding matrix.
  
  Once a dictionary D is found, the sparse coding model may be used to encode
  other matrices, and saved for future usage.
  
  To run this program, either an input matrix or an already-saved sparse coding
  model must be specified.  An input matrix may be specified with the "Training"
  option, along with the number of atoms in the dictionary (specified with the
  "Atoms" parameter).  It is also possible to specify an initial dictionary for
  the optimization, with the "InitialDictionary" parameter.  An input model may
  be specified with the "InputModel" parameter.

  As an example, to build a sparse coding model on the dataset data using 200
  atoms and an l1-regularization parameter of 0.1, saving the model into model,
  use 
  
  // Initialize optional parameters for SparseCoding().
  param := mlpack.SparseCodingOptions()
  param.Training = data
  param.Atoms = 200
  param.Lambda1 = 0.1
  
  _, _, model := mlpack.SparseCoding(param)
  
  Then, this model could be used to encode a new matrix, otherdata, and save the
  output codes to codes: 
  
  // Initialize optional parameters for SparseCoding().
  param := mlpack.SparseCodingOptions()
  param.InputModel = &model
  param.Test = otherdata
  
  codes, _, _ := mlpack.SparseCoding(param)

  Input parameters:

   - Atoms (int): Number of atoms in the dictionary.  Default value 15.
   - InitialDictionary (mat.Dense): Optional initial dictionary matrix.
   - InputModel (sparseCoding): File containing input sparse coding
        model.
   - Lambda1 (float64): Sparse coding l1-norm regularization parameter. 
        Default value 0.
   - Lambda2 (float64): Sparse coding l2-norm regularization parameter. 
        Default value 0.
   - MaxIterations (int): Maximum number of iterations for sparse coding
        (0 indicates no limit).  Default value 0.
   - NewtonTolerance (float64): Tolerance for convergence of Newton
        method.  Default value 1e-06.
   - Normalize (bool): If set, the input data matrix will be normalized
        before coding.
   - ObjectiveTolerance (float64): Tolerance for convergence of the
        objective function.  Default value 0.01.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - Test (mat.Dense): Optional matrix to be encoded by trained model.
   - Training (mat.Dense): Matrix of training data (X).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - codes (mat.Dense): Matrix to save the output sparse codes of the test
        matrix (--test_file) to.
   - dictionary (mat.Dense): Matrix to save the output dictionary to.
   - outputModel (sparseCoding): File to save trained sparse coding model
        to.

 */
func SparseCoding(param *SparseCodingOptionalParam) (*mat.Dense, *mat.Dense, sparseCoding) {
  params := getParams("sparse_coding")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.Atoms != 15 {
    setParamInt(params, "atoms", param.Atoms)
    setPassed(params, "atoms")
  }

  // Detect if the parameter was passed; set if so.
  if param.InitialDictionary != nil {
    gonumToArmaMat(params, "initial_dictionary", param.InitialDictionary, false)
    setPassed(params, "initial_dictionary")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setSparseCoding(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda1 != 0 {
    setParamDouble(params, "lambda1", param.Lambda1)
    setPassed(params, "lambda1")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda2 != 0 {
    setParamDouble(params, "lambda2", param.Lambda2)
    setPassed(params, "lambda2")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 0 {
    setParamInt(params, "max_iterations", param.MaxIterations)
    setPassed(params, "max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.NewtonTolerance != 1e-06 {
    setParamDouble(params, "newton_tolerance", param.NewtonTolerance)
    setPassed(params, "newton_tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalize != false {
    setParamBool(params, "normalize", param.Normalize)
    setPassed(params, "normalize")
  }

  // Detect if the parameter was passed; set if so.
  if param.ObjectiveTolerance != 0.01 {
    setParamDouble(params, "objective_tolerance", param.ObjectiveTolerance)
    setPassed(params, "objective_tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt(params, "seed", param.Seed)
    setPassed(params, "seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Test != nil {
    gonumToArmaMat(params, "test", param.Test, false)
    setPassed(params, "test")
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

  // Mark all output options as passed.
  setPassed(params, "codes")
  setPassed(params, "dictionary")
  setPassed(params, "output_model")

  // Call the mlpack program.
  C.mlpackSparseCoding(params.mem, timers.mem)

  // Initialize result variable and get output.
  var codesPtr mlpackArma
  codes := codesPtr.armaToGonumMat(params, "codes")
  var dictionaryPtr mlpackArma
  dictionary := dictionaryPtr.armaToGonumMat(params, "dictionary")
  var outputModel sparseCoding
  outputModel.getSparseCoding(params, "output_model")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return codes, dictionary, outputModel
}
