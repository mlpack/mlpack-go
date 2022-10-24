package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_test_go_binding
#include <capi/test_go_binding.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type TestGoBindingOptionalParam struct {
    BuildModel bool
    ColIn *mat.Dense
    Flag1 bool
    Flag2 bool
    MatrixAndInfoIn *matrixWithInfo
    MatrixIn *mat.Dense
    ModelIn *gaussianKernel
    RowIn *mat.Dense
    StrVectorIn []string
    UcolIn *mat.Dense
    UmatrixIn *mat.Dense
    UrowIn *mat.Dense
    VectorIn []int
    Verbose bool
}

func TestGoBindingOptions() *TestGoBindingOptionalParam {
  return &TestGoBindingOptionalParam{
    BuildModel: false,
    ColIn: nil,
    Flag1: false,
    Flag2: false,
    MatrixAndInfoIn: nil,
    MatrixIn: nil,
    ModelIn: nil,
    RowIn: nil,
    UcolIn: nil,
    UmatrixIn: nil,
    UrowIn: nil,
    Verbose: false,
  }
}

/*
  A simple program to test Go binding functionality.  You can build mlpack with
  the BUILD_TESTS option set to off, and this binding will no longer be built.

  Input parameters:

   - doubleIn (float64): Input double, must be 4.0.
   - intIn (int): Input int, must be 12.
   - stringIn (string): Input string, must be 'hello'.
   - BuildModel (bool): If true, a model will be returned.
   - ColIn (mat.Dense): Input column.
   - Flag1 (bool): Input flag, must be specified.
   - Flag2 (bool): Input flag, must not be specified.
   - MatrixAndInfoIn (matrixWithInfo): Input matrix and info.
   - MatrixIn (mat.Dense): Input matrix.
   - ModelIn (gaussianKernel): Input model.
   - RowIn (mat.Dense): Input row.
   - StrVectorIn ([]string): Input vector of strings.
   - UcolIn (mat.Dense): Input unsigned column.
   - UmatrixIn (mat.Dense): Input unsigned matrix.
   - UrowIn (mat.Dense): Input unsigned row.
   - VectorIn ([]int): Input vector of numbers.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - colOut (mat.Dense): Output column. 2x input column
   - doubleOut (float64): Output double, will be 5.0.  Default value 0.
   - intOut (int): Output int, will be 13.  Default value 0.
   - matrixAndInfoOut (mat.Dense): Output matrix and info; all numeric
        elements multiplied by 3.
   - matrixOut (mat.Dense): Output matrix.
   - modelBwOut (float64): The bandwidth of the model.  Default value 0.
   - modelOut (gaussianKernel): Output model, with twice the bandwidth.
   - rowOut (mat.Dense): Output row.  2x input row.
   - strVectorOut ([]string): Output string vector.
   - stringOut (string): Output string, will be 'hello2'.  Default value
        ''.
   - ucolOut (mat.Dense): Output unsigned column. 2x input column.
   - umatrixOut (mat.Dense): Output unsigned matrix.
   - urowOut (mat.Dense): Output unsigned row.  2x input row.
   - vectorOut ([]int): Output vector.

 */
func TestGoBinding(doubleIn float64, intIn int, stringIn string, param *TestGoBindingOptionalParam) (*mat.Dense, float64, int, *mat.Dense, *mat.Dense, float64, gaussianKernel, *mat.Dense, []string, string, *mat.Dense, *mat.Dense, *mat.Dense, []int) {
  params := getParams("test_go_binding")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  setParamDouble(params, "double_in", doubleIn)
  setPassed(params, "double_in")

  // Detect if the parameter was passed; set if so.
  setParamInt(params, "int_in", intIn)
  setPassed(params, "int_in")

  // Detect if the parameter was passed; set if so.
  setParamString(params, "string_in", stringIn)
  setPassed(params, "string_in")

  // Detect if the parameter was passed; set if so.
  if param.BuildModel != false {
    setParamBool(params, "build_model", param.BuildModel)
    setPassed(params, "build_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.ColIn != nil {
    gonumToArmaCol(params, "col_in", param.ColIn)
    setPassed(params, "col_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Flag1 != false {
    setParamBool(params, "flag1", param.Flag1)
    setPassed(params, "flag1")
  }

  // Detect if the parameter was passed; set if so.
  if param.Flag2 != false {
    setParamBool(params, "flag2", param.Flag2)
    setPassed(params, "flag2")
  }

  // Detect if the parameter was passed; set if so.
  if param.MatrixAndInfoIn != nil {
    gonumToArmaMatWithInfo(params, "matrix_and_info_in", param.MatrixAndInfoIn)
    setPassed(params, "matrix_and_info_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.MatrixIn != nil {
    gonumToArmaMat(params, "matrix_in", param.MatrixIn)
    setPassed(params, "matrix_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.ModelIn != nil {
    setGaussianKernel(params, "model_in", param.ModelIn)
    setPassed(params, "model_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.RowIn != nil {
    gonumToArmaRow(params, "row_in", param.RowIn)
    setPassed(params, "row_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.StrVectorIn != nil {
    setParamVecString(params, "str_vector_in", param.StrVectorIn)
    setPassed(params, "str_vector_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.UcolIn != nil {
    gonumToArmaUcol(params, "ucol_in", param.UcolIn)
    setPassed(params, "ucol_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.UmatrixIn != nil {
    gonumToArmaUmat(params, "umatrix_in", param.UmatrixIn)
    setPassed(params, "umatrix_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.UrowIn != nil {
    gonumToArmaUrow(params, "urow_in", param.UrowIn)
    setPassed(params, "urow_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.VectorIn != nil {
    setParamVecInt(params, "vector_in", param.VectorIn)
    setPassed(params, "vector_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "col_out")
  setPassed(params, "double_out")
  setPassed(params, "int_out")
  setPassed(params, "matrix_and_info_out")
  setPassed(params, "matrix_out")
  setPassed(params, "model_bw_out")
  setPassed(params, "model_out")
  setPassed(params, "row_out")
  setPassed(params, "str_vector_out")
  setPassed(params, "string_out")
  setPassed(params, "ucol_out")
  setPassed(params, "umatrix_out")
  setPassed(params, "urow_out")
  setPassed(params, "vector_out")

  // Call the mlpack program.
  C.mlpackTestGoBinding(params.mem, timers.mem)

  // Initialize result variable and get output.
  var colOutPtr mlpackArma
  colOut := colOutPtr.armaToGonumCol(params, "col_out")
  doubleOut := getParamDouble(params, "double_out")
  intOut := getParamInt(params, "int_out")
  var matrixAndInfoOutPtr mlpackArma
  matrixAndInfoOut := matrixAndInfoOutPtr.armaToGonumMat(params, "matrix_and_info_out")
  var matrixOutPtr mlpackArma
  matrixOut := matrixOutPtr.armaToGonumMat(params, "matrix_out")
  modelBwOut := getParamDouble(params, "model_bw_out")
  var modelOut gaussianKernel
  modelOut.getGaussianKernel(params, "model_out")
  var rowOutPtr mlpackArma
  rowOut := rowOutPtr.armaToGonumRow(params, "row_out")
  strVectorOut := getParamVecString(params, "str_vector_out")
  stringOut := getParamString(params, "string_out")
  var ucolOutPtr mlpackArma
  ucolOut := ucolOutPtr.armaToGonumUcol(params, "ucol_out")
  var umatrixOutPtr mlpackArma
  umatrixOut := umatrixOutPtr.armaToGonumUmat(params, "umatrix_out")
  var urowOutPtr mlpackArma
  urowOut := urowOutPtr.armaToGonumUrow(params, "urow_out")
  vectorOut := getParamVecInt(params, "vector_out")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return colOut, doubleOut, intOut, matrixAndInfoOut, matrixOut, modelBwOut, modelOut, rowOut, strVectorOut, stringOut, ucolOut, umatrixOut, urowOut, vectorOut
}
