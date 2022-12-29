package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_kmeans
#include <capi/kmeans.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type KmeansOptionalParam struct {
    Algorithm string
    AllowEmptyClusters bool
    InPlace bool
    InitialCentroids *mat.Dense
    KillEmptyClusters bool
    KmeansPlusPlus bool
    LabelsOnly bool
    MaxIterations int
    Percentage float64
    RefinedStart bool
    Samplings int
    Seed int
    Verbose bool
}

func KmeansOptions() *KmeansOptionalParam {
  return &KmeansOptionalParam{
    Algorithm: "naive",
    AllowEmptyClusters: false,
    InPlace: false,
    InitialCentroids: nil,
    KillEmptyClusters: false,
    KmeansPlusPlus: false,
    LabelsOnly: false,
    MaxIterations: 1000,
    Percentage: 0.02,
    RefinedStart: false,
    Samplings: 100,
    Seed: 0,
    Verbose: false,
  }
}

/*
  This program performs K-Means clustering on the given dataset.  It can return
  the learned cluster assignments, and the centroids of the clusters.  Empty
  clusters are not allowed by default; when a cluster becomes empty, the point
  furthest from the centroid of the cluster with maximum variance is taken to
  fill that cluster.
  
  Optionally, the strategy to choose initial centroids can be specified.  The
  k-means++ algorithm can be used to choose initial centroids with the
  "KmeansPlusPlus" parameter.  The Bradley and Fayyad approach ("Refining
  initial points for k-means clustering", 1998) can be used to select initial
  points by specifying the "RefinedStart" parameter.  This approach works by
  taking random samplings of the dataset; to specify the number of samplings,
  the "Samplings" parameter is used, and to specify the percentage of the
  dataset to be used in each sample, the "Percentage" parameter is used (it
  should be a value between 0.0 and 1.0).
  
  There are several options available for the algorithm used for each Lloyd
  iteration, specified with the "Algorithm"  option.  The standard O(kN)
  approach can be used ('naive').  Other options include the Pelleg-Moore
  tree-based algorithm ('pelleg-moore'), Elkan's triangle-inequality based
  algorithm ('elkan'), Hamerly's modification to Elkan's algorithm ('hamerly'),
  the dual-tree k-means algorithm ('dualtree'), and the dual-tree k-means
  algorithm using the cover tree ('dualtree-covertree').
  
  The behavior for when an empty cluster is encountered can be modified with the
  "AllowEmptyClusters" option.  When this option is specified and there is a
  cluster owning no points at the end of an iteration, that cluster's centroid
  will simply remain in its position from the previous iteration. If the
  "KillEmptyClusters" option is specified, then when a cluster owns no points at
  the end of an iteration, the cluster centroid is simply filled with DBL_MAX,
  killing it and effectively reducing k for the rest of the computation.  Note
  that the default option when neither empty cluster option is specified can be
  time-consuming to calculate; therefore, specifying either of these parameters
  will often accelerate runtime.
  
  Initial clustering assignments may be specified using the "InitialCentroids"
  parameter, and the maximum number of iterations may be specified with the
  "MaxIterations" parameter.

  As an example, to use Hamerly's algorithm to perform k-means clustering with
  k=10 on the dataset data, saving the centroids to centroids and the
  assignments for each point to assignments, the following command could be
  used:
  
  // Initialize optional parameters for Kmeans().
  param := mlpack.KmeansOptions()
  
  centroids, assignments := mlpack.Kmeans(data, 10, param)
  
  To run k-means on that same dataset with initial centroids specified in
  initial with a maximum of 500 iterations, storing the output centroids in
  final the following command may be used:
  
  // Initialize optional parameters for Kmeans().
  param := mlpack.KmeansOptions()
  param.InitialCentroids = initial
  param.MaxIterations = 500
  
  final, _ := mlpack.Kmeans(data, 10, param)

  Input parameters:

   - clusters (int): Number of clusters to find (0 autodetects from
        initial centroids).
   - input (mat.Dense): Input dataset to perform clustering on.
   - Algorithm (string): Algorithm to use for the Lloyd iteration
        ('naive', 'pelleg-moore', 'elkan', 'hamerly', 'dualtree', or
        'dualtree-covertree').  Default value 'naive'.
   - AllowEmptyClusters (bool): Allow empty clusters to be persist.
   - InPlace (bool): If specified, a column containing the learned cluster
        assignments will be added to the input dataset file.  In this case,
        --output_file is overridden. (Do not use in Python.)
   - InitialCentroids (mat.Dense): Start with the specified initial
        centroids.
   - KillEmptyClusters (bool): Remove empty clusters when they occur.
   - KmeansPlusPlus (bool): Use the k-means++ initialization strategy to
        choose initial points.
   - LabelsOnly (bool): Only output labels into output file.
   - MaxIterations (int): Maximum number of iterations before k-means
        terminates.  Default value 1000.
   - Percentage (float64): Percentage of dataset to use for each refined
        start sampling (use when --refined_start is specified).  Default value
        0.02.
   - RefinedStart (bool): Use the refined initial point strategy by
        Bradley and Fayyad to choose initial points.
   - Samplings (int): Number of samplings to perform for refined start
        (use when --refined_start is specified).  Default value 100.
   - Seed (int): Random seed.  If 0, 'std::time(NULL)' is used.  Default
        value 0.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - centroid (mat.Dense): If specified, the centroids of each cluster
        will  be written to the given file.
   - output (mat.Dense): Matrix to store output labels or labeled data
        to.

 */
func Kmeans(clusters int, input *mat.Dense, param *KmeansOptionalParam) (*mat.Dense, *mat.Dense) {
  params := getParams("kmeans")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  setParamInt(params, "clusters", clusters)
  setPassed(params, "clusters")

  // Detect if the parameter was passed; set if so.
  gonumToArmaMat(params, "input", input, false)
  setPassed(params, "input")

  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "naive" {
    setParamString(params, "algorithm", param.Algorithm)
    setPassed(params, "algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.AllowEmptyClusters != false {
    setParamBool(params, "allow_empty_clusters", param.AllowEmptyClusters)
    setPassed(params, "allow_empty_clusters")
  }

  // Detect if the parameter was passed; set if so.
  if param.InPlace != false {
    setParamBool(params, "in_place", param.InPlace)
    setPassed(params, "in_place")
  }

  // Detect if the parameter was passed; set if so.
  if param.InitialCentroids != nil {
    gonumToArmaMat(params, "initial_centroids", param.InitialCentroids, false)
    setPassed(params, "initial_centroids")
  }

  // Detect if the parameter was passed; set if so.
  if param.KillEmptyClusters != false {
    setParamBool(params, "kill_empty_clusters", param.KillEmptyClusters)
    setPassed(params, "kill_empty_clusters")
  }

  // Detect if the parameter was passed; set if so.
  if param.KmeansPlusPlus != false {
    setParamBool(params, "kmeans_plus_plus", param.KmeansPlusPlus)
    setPassed(params, "kmeans_plus_plus")
  }

  // Detect if the parameter was passed; set if so.
  if param.LabelsOnly != false {
    setParamBool(params, "labels_only", param.LabelsOnly)
    setPassed(params, "labels_only")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 1000 {
    setParamInt(params, "max_iterations", param.MaxIterations)
    setPassed(params, "max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.Percentage != 0.02 {
    setParamDouble(params, "percentage", param.Percentage)
    setPassed(params, "percentage")
  }

  // Detect if the parameter was passed; set if so.
  if param.RefinedStart != false {
    setParamBool(params, "refined_start", param.RefinedStart)
    setPassed(params, "refined_start")
  }

  // Detect if the parameter was passed; set if so.
  if param.Samplings != 100 {
    setParamInt(params, "samplings", param.Samplings)
    setPassed(params, "samplings")
  }

  // Detect if the parameter was passed; set if so.
  if param.Seed != 0 {
    setParamInt(params, "seed", param.Seed)
    setPassed(params, "seed")
  }

  // Detect if the parameter was passed; set if so.
  if param.Verbose != false {
    setParamBool(params, "verbose", param.Verbose)
    setPassed(params, "verbose")
    enableVerbose()
  }

  // Mark all output options as passed.
  setPassed(params, "centroid")
  setPassed(params, "output")

  // Call the mlpack program.
  C.mlpackKmeans(params.mem, timers.mem)

  // Initialize result variable and get output.
  var centroidPtr mlpackArma
  centroid := centroidPtr.armaToGonumMat(params, "centroid")
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumMat(params, "output")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return centroid, output
}
