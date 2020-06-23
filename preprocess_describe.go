package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_preprocess_describe
#include <capi/preprocess_describe.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type PreprocessDescribeOptionalParam struct {
    Dimension int
    Population bool
    Precision int
    RowMajor bool
    Verbose bool
    Width int
}

func PreprocessDescribeOptions() *PreprocessDescribeOptionalParam {
  return &PreprocessDescribeOptionalParam{
    Dimension: 0,
    Population: false,
    Precision: 4,
    RowMajor: false,
    Verbose: false,
    Width: 8,
  }
}

/*
  This utility takes a dataset and prints out the descriptive statistics of the
  data. Descriptive statistics is the discipline of quantitatively describing
  the main features of a collection of information, or the quantitative
  description itself. The program does not modify the original file, but instead
  prints out the statistics to the console. The printed result will look like a
  table.
  
  Optionally, width and precision of the output can be adjusted by a user using
  the "Width" and "Precision" parameters. A user can also select a specific
  dimension to analyze if there are too many dimensions. The "Population"
  parameter can be specified when the dataset should be considered as a
  population.  Otherwise, the dataset will be considered as a sample.
  
  So, a simple example where we want to print out statistical facts about the
  dataset X using the default settings, we could run 
  
  // Initialize optional parameters for PreprocessDescribe().
  param := mlpack.PreprocessDescribeOptions()
  param.Verbose = true
  
   := mlpack.PreprocessDescribe(X, param)
  
  If we want to customize the width to 10 and precision to 5 and consider the
  dataset as a population, we could run
  
  // Initialize optional parameters for PreprocessDescribe().
  param := mlpack.PreprocessDescribeOptions()
  param.Width = 10
  param.Precision = 5
  param.Verbose = true
  
   := mlpack.PreprocessDescribe(X, param)


  Input parameters:

   - input (mat.Dense): Matrix containing data,
   - Dimension (int): Dimension of the data. Use this to specify a
        dimension  Default value 0.
   - Population (bool): If specified, the program will calculate
        statistics assuming the dataset is the population. By default, the
        program will assume the dataset as a sample.
   - Precision (int): Precision of the output statistics.  Default value
        4.
   - RowMajor (bool): If specified, the program will calculate statistics
        across rows, not across columns.  (Remember that in mlpack, a column
        represents a point, so this option is generally not necessary.)
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.
   - Width (int): Width of the output table.  Default value 8.

  Output parameters:


 */
func PreprocessDescribe(input *mat.Dense, param *PreprocessDescribeOptionalParam) () {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Descriptive Statistics")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Dimension != 0 {
    setParamInt("dimension", param.Dimension)
    setPassed("dimension")
  }

  // Detect if the parameter was passed; set if so.
  if param.Population != false {
    setParamBool("population", param.Population)
    setPassed("population")
  }

  // Detect if the parameter was passed; set if so.
  if param.Precision != 4 {
    setParamInt("precision", param.Precision)
    setPassed("precision")
  }

  // Detect if the parameter was passed; set if so.
  if param.RowMajor != false {
    setParamBool("row_major", param.RowMajor)
    setPassed("row_major")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Detect if the parameter was passed; set if so.
  if param.Width != 8 {
    setParamInt("width", param.Width)
    setPassed("width")
  }

  // Mark all output options as passed.

  // Call the mlpack program.
  C.mlpackPreprocessDescribe()

  // Initialize result variable and get output.

  // Clear settings.
  clearSettings()

  // Return output(s).
  return 
}
