package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_local_coordinate_coding
#include <capi/local_coordinate_coding.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type LocalCoordinateCodingOptionalParam struct {
    Atoms int
    InitialDictionary *mat.Dense
    InputModel *localCoordinateCoding
    Lambda float64
    MaxIterations int
    Normalize bool
    Seed int
    Test *mat.Dense
    Tolerance float64
    Training *mat.Dense
    Verbose bool
}

func LocalCoordinateCodingOptions() *LocalCoordinateCodingOptionalParam {
  return &LocalCoordinateCodingOptionalParam{
    Atoms: 0,
    InitialDictionary: nil,
    InputModel: nil,
    Lambda: 0,
    MaxIterations: 0,
    Normalize: false,
    Seed: 0,
    Test: nil,
    Tolerance: 0.01,
    Training: nil,
    Verbose: false,
  }
}

/*
  An implementation of Local Coordinate Coding (LCC), which codes data that
  approximately lives on a manifold using a variation of l1-norm regularized
  sparse coding.  Given a dense data matrix X with n points and d dimensions,
  LCC seeks to find a dense dictionary matrix D with k atoms in d dimensions,
  and a coding matrix Z with n points in k dimensions.  Because of the
  regularization method used, the atoms in D should lie close to the manifold on
  which the data points lie.
  
  The original data matrix X can then be reconstructed as D * Z.  Therefore,
  this program finds a representation of each point in X as a sparse linear
  combination of atoms in the dictionary D.
  
  The coding is found with an algorithm which alternates between a dictionary
  step, which updates the dictionary D, and a coding step, which updates the
  coding matrix Z.
  
  To run this program, the input matrix X must be specified (with -i), along
  with the number of atoms in the dictionary (-k).  An initial dictionary may
  also be specified with the "InitialDictionary" parameter.  The l1-norm
  regularization parameter is specified with the "Lambda" parameter.

  For example, to run LCC on the dataset data using 200 atoms and an
  l1-regularization parameter of 0.1, saving the dictionary "Dictionary" and the
  codes into "Codes", use
  
  // Initialize optional parameters for LocalCoordinateCoding().
  param := mlpack.LocalCoordinateCodingOptions()
  param.Training = data
  param.Atoms = 200
  param.Lambda = 0.1
  
  codes, dict, _ := mlpack.LocalCoordinateCoding(param)
  
  The maximum number of iterations may be specified with the "MaxIterations"
  parameter. Optionally, the input data matrix X can be normalized before coding
  with the "Normalize" parameter.
  
  An LCC model may be saved using the "OutputModel" output parameter.  Then, to
  encode new points from the dataset points with the previously saved model
  lcc_model, saving the new codes to new_codes, the following command can be
  used:
  
  // Initialize optional parameters for LocalCoordinateCoding().
  param := mlpack.LocalCoordinateCodingOptions()
  param.InputModel = &lcc_model
  param.Test = points
  
  new_codes, _, _ := mlpack.LocalCoordinateCoding(param)

  Input parameters:

   - Atoms (int): Number of atoms in the dictionary.  Default value 0.
   - InitialDictionary (mat.Dense): Optional initial dictionary.
   - InputModel (localCoordinateCoding): Input LCC model.
   - Lambda (float64): Weighted l1-norm regularization parameter.  Default
        value 0.
   - MaxIterations (int): Maximum number of iterations for LCC (0
        indicates no limit).  Default value 0.
   - Normalize (bool): If set, the input data matrix will be normalized
        before coding.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - Test (mat.Dense): Test points to encode.
   - Tolerance (float64): Tolerance for objective function.  Default value
        0.01.
   - Training (mat.Dense): Matrix of training data (X).
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - codes (mat.Dense): Output codes matrix.
   - dictionary (mat.Dense): Output dictionary matrix.
   - outputModel (localCoordinateCoding): Output for trained LCC model.

 */
func LocalCoordinateCoding(param *LocalCoordinateCodingOptionalParam) (*mat.Dense, *mat.Dense, localCoordinateCoding) {
  params := getParams("local_coordinate_coding")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.Atoms != 0 {
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
    setLocalCoordinateCoding(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Lambda != 0 {
    setParamDouble(params, "lambda", param.Lambda)
    setPassed(params, "lambda")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 0 {
    setParamInt(params, "max_iterations", param.MaxIterations)
    setPassed(params, "max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalize != false {
    setParamBool(params, "normalize", param.Normalize)
    setPassed(params, "normalize")
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
  if param.Tolerance != 0.01 {
    setParamDouble(params, "tolerance", param.Tolerance)
    setPassed(params, "tolerance")
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
  C.mlpackLocalCoordinateCoding(params.mem, timers.mem)

  // Initialize result variable and get output.
  var codesPtr mlpackArma
  codes := codesPtr.armaToGonumMat(params, "codes")
  var dictionaryPtr mlpackArma
  dictionary := dictionaryPtr.armaToGonumMat(params, "dictionary")
  var outputModel localCoordinateCoding
  outputModel.getLocalCoordinateCoding(params, "output_model")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return codes, dictionary, outputModel
}
