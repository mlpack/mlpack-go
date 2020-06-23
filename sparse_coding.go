package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_sparse_coding
#include <capi/sparse_coding.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

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

type sparseCoding struct {
  mem unsafe.Pointer
}

func (m *sparseCoding) allocSparseCoding(identifier string) {
  m.mem = C.mlpackGetSparseCodingPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *sparseCoding) getSparseCoding(identifier string) {
  m.allocSparseCoding(identifier)
}

func setSparseCoding(identifier string, ptr *sparseCoding) {
  C.mlpackSetSparseCodingPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
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
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Sparse Coding")

  // Detect if the parameter was passed; set if so.
  if param.Atoms != 15 {
    setParamInt("atoms", param.Atoms)
    setPassed("atoms")
  }

  // Detect if the parameter was passed; set if so.
  if param.InitialDictionary != nil {
    gonumToArmaMat("initial_dictionary", param.InitialDictionary)
    setPassed("initial_dictionary")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setSparseCoding("input_model", param.InputModel)
    setPassed("input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda1 != 0 {
    setParamDouble("lambda1", param.Lambda1)
    setPassed("lambda1")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda2 != 0 {
    setParamDouble("lambda2", param.Lambda2)
    setPassed("lambda2")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 0 {
    setParamInt("max_iterations", param.MaxIterations)
    setPassed("max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.NewtonTolerance != 1e-06 {
    setParamDouble("newton_tolerance", param.NewtonTolerance)
    setPassed("newton_tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalize != false {
    setParamBool("normalize", param.Normalize)
    setPassed("normalize")
  }

  // Detect if the parameter was passed; set if so.
  if param.ObjectiveTolerance != 0.01 {
    setParamDouble("objective_tolerance", param.ObjectiveTolerance)
    setPassed("objective_tolerance")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt("seed", param.Seed)
    setPassed("seed")
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
  setPassed("codes")
  setPassed("dictionary")
  setPassed("output_model")

  // Call the mlpack program.
  C.mlpackSparseCoding()

  // Initialize result variable and get output.
  var codesPtr mlpackArma
  codes := codesPtr.armaToGonumMat("codes")
  var dictionaryPtr mlpackArma
  dictionary := dictionaryPtr.armaToGonumMat("dictionary")
  var outputModel sparseCoding
  outputModel.getSparseCoding("output_model")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return codes, dictionary, outputModel
}
