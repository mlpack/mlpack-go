package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_image_converter
#include <capi/image_converter.h>
#include <stdlib.h>
*/
import "C" 

import (
  "gonum.org/v1/gonum/mat" 
)

type ImageConverterOptionalParam struct {
    Channels int
    Dataset *mat.Dense
    Height int
    Quality int
    Save bool
    Verbose bool
    Width int
}

func ImageConverterOptions() *ImageConverterOptionalParam {
  return &ImageConverterOptionalParam{
    Channels: 0,
    Dataset: nil,
    Height: 0,
    Quality: 90,
    Save: false,
    Verbose: false,
    Width: 0,
  }
}

/*
  This utility takes an image or an array of images and loads them to a matrix.
  You can optionally specify the height "Height" width "Width" and channel
  "Channels" of the images that needs to be loaded; otherwise, these parameters
  will be automatically detected from the image.
  There are other options too, that can be specified such as "Quality".
  
  You can also provide a dataset and save them as images using "Dataset" and
  "Save" as an parameter. An example to load an image : 
  
  // Initialize optional parameters for ImageConverter().
  param := mlpack.ImageConverterOptions()
  param.Height = 256
  param.Width = 256
  param.Channels = 3
  
  Y := mlpack.ImageConverter(X, param)
  
   An example to save an image is :
  
  // Initialize optional parameters for ImageConverter().
  param := mlpack.ImageConverterOptions()
  param.Height = 256
  param.Width = 256
  param.Channels = 3
  param.Dataset = Y
  param.Save = true
  
  _ := mlpack.ImageConverter(X, param)


  Input parameters:

   - input ([]string): Image filenames which have to be loaded/saved.
   - Channels (int): Number of channels in the image.  Default value 0.
   - Dataset (mat.Dense): Input matrix to save as images.
   - Height (int): Height of the images.  Default value 0.
   - Quality (int): Compression of the image if saved as jpg (0-100). 
        Default value 90.
   - Save (bool): Save a dataset as images.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.
   - Width (int): Width of the image.  Default value 0.

  Output parameters:

   - output (mat.Dense): Matrix to save images data to, Onlyneeded if you
        are specifying 'save' option.

 */
func ImageConverter(input []string, param *ImageConverterOptionalParam) (*mat.Dense) {
  resetTimers()
  enableTimers()
  disableBacktrace()
  disableVerbose()
  restoreSettings("Image Converter")

  // Detect if the parameter was passed; set if so.
  setParamVecString("input", input)
  setPassed("input")

  // Detect if the parameter was passed; set if so.
  if param.Channels != 0 {
    setParamInt("channels", param.Channels)
    setPassed("channels")
  }

  // Detect if the parameter was passed; set if so.
  if param.Dataset != nil {
    gonumToArmaMat("dataset", param.Dataset)
    setPassed("dataset")
  }

  // Detect if the parameter was passed; set if so.
  if param.Height != 0 {
    setParamInt("height", param.Height)
    setPassed("height")
  }

  // Detect if the parameter was passed; set if so.
  if param.Quality != 90 {
    setParamInt("quality", param.Quality)
    setPassed("quality")
  }

  // Detect if the parameter was passed; set if so.
  if param.Save != false {
    setParamBool("save", param.Save)
    setPassed("save")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool("verbose", param.Verbose)
    setPassed("verbose")
    enableVerbose()
  }

  // Detect if the parameter was passed; set if so.
  if param.Width != 0 {
    setParamInt("width", param.Width)
    setPassed("width")
  }

  // Mark all output options as passed.
  setPassed("output")

  // Call the mlpack program.
  C.mlpackImageConverter()

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat("output")

  // Clear settings.
  clearSettings()

  // Return output(s).
  return output
}
