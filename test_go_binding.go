package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_test_go_binding
#include <capi/test_go_binding.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
  "runtime" 
  "unsafe" 
)

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

type gaussianKernel struct {
  mem unsafe.Pointer
}

func (m *gaussianKernel) allocGaussianKernel(identifier string) {
  m.mem = C.mlpackGetGaussianKernelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *gaussianKernel) getGaussianKernel(identifier string) {
  m.allocGaussianKernel(identifier)
}

func setGaussianKernel(identifier string, ptr *gaussianKernel) {
  C.mlpackSetGaussianKernelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

/*
  A simple program to test Golang binding functionality.  You can build mlpack
  with the BUILD_TESTS option set to off, and this binding will no longer be
  built.


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
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Golang binding test")

  // Detect if the parameter was passed; set if so.
  setParamDouble("double_in", doubleIn)
  setPassed("double_in")

  // Detect if the parameter was passed; set if so.
  setParamInt("int_in", intIn)
  setPassed("int_in")

  // Detect if the parameter was passed; set if so.
  setParamString("string_in", stringIn)
  setPassed("string_in")

  // Detect if the parameter was passed; set if so.
  if param.BuildModel != false {
    setParamBool("build_model", param.BuildModel)
    setPassed("build_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.ColIn != nil {
    gonumToArmaCol("col_in", param.ColIn)
    setPassed("col_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Flag1 != false {
    setParamBool("flag1", param.Flag1)
    setPassed("flag1")
  }

  // Detect if the parameter was passed; set if so.
  if param.Flag2 != false {
    setParamBool("flag2", param.Flag2)
    setPassed("flag2")
  }

  // Detect if the parameter was passed; set if so.
  if param.MatrixAndInfoIn != nil {
    gonumToArmaMatWithInfo("matrix_and_info_in", param.MatrixAndInfoIn)
    setPassed("matrix_and_info_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.MatrixIn != nil {
    gonumToArmaMat("matrix_in", param.MatrixIn)
    setPassed("matrix_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.ModelIn != nil {
    setGaussianKernel("model_in", param.ModelIn)
    setPassed("model_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.RowIn != nil {
    gonumToArmaRow("row_in", param.RowIn)
    setPassed("row_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.StrVectorIn != nil {
    setParamVecString("str_vector_in", param.StrVectorIn)
    setPassed("str_vector_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.UcolIn != nil {
    gonumToArmaUcol("ucol_in", param.UcolIn)
    setPassed("ucol_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.UmatrixIn != nil {
    gonumToArmaUmat("umatrix_in", param.UmatrixIn)
    setPassed("umatrix_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.UrowIn != nil {
    gonumToArmaUrow("urow_in", param.UrowIn)
    setPassed("urow_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.VectorIn != nil {
    setParamVecInt("vector_in", param.VectorIn)
    setPassed("vector_in")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed("col_out")
  setPassed("double_out")
  setPassed("int_out")
  setPassed("matrix_and_info_out")
  setPassed("matrix_out")
  setPassed("model_bw_out")
  setPassed("model_out")
  setPassed("row_out")
  setPassed("str_vector_out")
  setPassed("string_out")
  setPassed("ucol_out")
  setPassed("umatrix_out")
  setPassed("urow_out")
  setPassed("vector_out")

  // Call the mlpack program.
  C.mlpackTestGoBinding()

  // Initialize result variable and get output.
  var colOutPtr mlpackArma
  colOut := colOutPtr.armaToGonumCol("col_out")
  doubleOut := getParamDouble("double_out")
  intOut := getParamInt("int_out")
  var matrixAndInfoOutPtr mlpackArma
  matrixAndInfoOut := matrixAndInfoOutPtr.armaToGonumMat("matrix_and_info_out")
  var matrixOutPtr mlpackArma
  matrixOut := matrixOutPtr.armaToGonumMat("matrix_out")
  modelBwOut := getParamDouble("model_bw_out")
  var modelOut gaussianKernel
  modelOut.getGaussianKernel("model_out")
  var rowOutPtr mlpackArma
  rowOut := rowOutPtr.armaToGonumRow("row_out")
  strVectorOut := getParamVecString("str_vector_out")
  stringOut := getParamString("string_out")
  var ucolOutPtr mlpackArma
  ucolOut := ucolOutPtr.armaToGonumUcol("ucol_out")
  var umatrixOutPtr mlpackArma
  umatrixOut := umatrixOutPtr.armaToGonumUmat("umatrix_out")
  var urowOutPtr mlpackArma
  urowOut := urowOutPtr.armaToGonumUrow("urow_out")
  vectorOut := getParamVecInt("vector_out")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return colOut, doubleOut, intOut, matrixAndInfoOut, matrixOut, modelBwOut, modelOut, rowOut, strVectorOut, stringOut, ucolOut, umatrixOut, urowOut, vectorOut
}
