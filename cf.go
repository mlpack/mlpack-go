package mlpack

/*
#cgo CFLAGS: -I./capi -Wall
#cgo LDFLAGS: -L. -lmlpack_go_cf
#include <capi/cf.h>
#include <stdlib.h>
*/
import "C" 

import "gonum.org/v1/gonum/mat" 

type CfOptionalParam struct {
    Algorithm string
    AllUserRecommendations bool
    InputModel *cfModel
    Interpolation string
    IterationOnlyTermination bool
    MaxIterations int
    MinResidue float64
    NeighborSearch string
    Neighborhood int
    Normalization string
    Query *mat.Dense
    Rank int
    Recommendations int
    Seed int
    Test *mat.Dense
    Training *mat.Dense
    Verbose bool
}

func CfOptions() *CfOptionalParam {
  return &CfOptionalParam{
    Algorithm: "NMF",
    AllUserRecommendations: false,
    InputModel: nil,
    Interpolation: "average",
    IterationOnlyTermination: false,
    MaxIterations: 1000,
    MinResidue: 1e-05,
    NeighborSearch: "euclidean",
    Neighborhood: 5,
    Normalization: "none",
    Query: nil,
    Rank: 0,
    Recommendations: 5,
    Seed: 0,
    Test: nil,
    Training: nil,
    Verbose: false,
  }
}

/*
  This program performs collaborative filtering (CF) on the given dataset. Given
  a list of user, item and preferences (the "Training" parameter), the program
  will perform a matrix decomposition and then can perform a series of actions
  related to collaborative filtering.  Alternately, the program can load an
  existing saved CF model with the "InputModel" parameter and then use that
  model to provide recommendations or predict values.
  
  The input matrix should be a 3-dimensional matrix of ratings, where the first
  dimension is the user, the second dimension is the item, and the third
  dimension is that user's rating of that item.  Both the users and items should
  be numeric indices, not names. The indices are assumed to start from 0.
  
  A set of query users for which recommendations can be generated may be
  specified with the "Query" parameter; alternately, recommendations may be
  generated for every user in the dataset by specifying the
  "AllUserRecommendations" parameter.  In addition, the number of
  recommendations per user to generate can be specified with the
  "Recommendations" parameter, and the number of similar users (the size of the
  neighborhood) to be considered when generating recommendations can be
  specified with the "Neighborhood" parameter.
  
  For performing the matrix decomposition, the following optimization algorithms
  can be specified via the "Algorithm" parameter: 
   - 'RegSVD' -- Regularized SVD using a SGD optimizer
   - 'NMF' -- Non-negative matrix factorization with alternating least squares
  update rules
   - 'BatchSVD' -- SVD batch learning
   - 'SVDIncompleteIncremental' -- SVD incomplete incremental learning
   - 'SVDCompleteIncremental' -- SVD complete incremental learning
   - 'BiasSVD' -- Bias SVD using a SGD optimizer
   - 'SVDPP' -- SVD++ using a SGD optimizer
  
  
  The following neighbor search algorithms can be specified via the
  "NeighborSearch" parameter:
   - 'cosine'  -- Cosine Search Algorithm
   - 'euclidean'  -- Euclidean Search Algorithm
   - 'pearson'  -- Pearson Search Algorithm
  
  
  The following weight interpolation algorithms can be specified via the
  "Interpolation" parameter:
   - 'average'  -- Average Interpolation Algorithm
   - 'regression'  -- Regression Interpolation Algorithm
   - 'similarity'  -- Similarity Interpolation Algorithm
  
  
  The following ranking normalization algorithms can be specified via the
  "Normalization" parameter:
   - 'none'  -- No Normalization
   - 'item_mean'  -- Item Mean Normalization
   - 'overall_mean'  -- Overall Mean Normalization
   - 'user_mean'  -- User Mean Normalization
   - 'z_score'  -- Z-Score Normalization
  
  A trained model may be saved to with the "OutputModel" output parameter.

  To train a CF model on a dataset training_set using NMF for decomposition and
  saving the trained model to model, one could call: 
  
  // Initialize optional parameters for Cf().
  param := mlpack.CfOptions()
  param.Training = training_set
  param.Algorithm = "NMF"
  
  _, model := mlpack.Cf(param)
  
  Then, to use this model to generate recommendations for the list of users in
  the query set users, storing 5 recommendations in recommendations, one could
  call 
  
  // Initialize optional parameters for Cf().
  param := mlpack.CfOptions()
  param.InputModel = &model
  param.Query = users
  param.Recommendations = 5
  
  recommendations, _ := mlpack.Cf(param)

  Input parameters:

   - Algorithm (string): Algorithm used for matrix factorization.  Default
        value 'NMF'.
   - AllUserRecommendations (bool): Generate recommendations for all
        users.
   - InputModel (cfModel): Trained CF model to load.
   - Interpolation (string): Algorithm used for weight interpolation. 
        Default value 'average'.
   - IterationOnlyTermination (bool): Terminate only when the maximum
        number of iterations is reached.
   - MaxIterations (int): Maximum number of iterations. If set to zero,
        there is no limit on the number of iterations.  Default value 1000.
   - MinResidue (float64): Residue required to terminate the factorization
        (lower values generally mean better fits).  Default value 1e-05.
   - NeighborSearch (string): Algorithm used for neighbor search.  Default
        value 'euclidean'.
   - Neighborhood (int): Size of the neighborhood of similar users to
        consider for each query user.  Default value 5.
   - Normalization (string): Normalization performed on the ratings. 
        Default value 'none'.
   - Query (mat.Dense): List of query users for which recommendations
        should be generated.
   - Rank (int): Rank of decomposed matrices (if 0, a heuristic is used to
        estimate the rank).  Default value 0.
   - Recommendations (int): Number of recommendations to generate for each
        query user.  Default value 5.
   - Seed (int): Set the random seed (0 uses std::time(NULL)).  Default
        value 0.
   - Test (mat.Dense): Test set to calculate RMSE on.
   - Training (mat.Dense): Input dataset to perform CF on.
   - Verbose (bool): Display informational messages and the full list of
        parameters and timers at the end of execution.

  Output parameters:

   - output (mat.Dense): Matrix that will store output recommendations.
   - outputModel (cfModel): Output for trained CF model.

 */
func Cf(param *CfOptionalParam) (*mat.Dense, cfModel) {
  params := getParams("cf")
  timers := getTimers()

  disableBacktrace()
  disableVerbose()
  // Detect if the parameter was passed; set if so.
  if param.Algorithm != "NMF" {
    setParamString(params, "algorithm", param.Algorithm)
    setPassed(params, "algorithm")
  }

  // Detect if the parameter was passed; set if so.
  if param.AllUserRecommendations != false {
    setParamBool(params, "all_user_recommendations", param.AllUserRecommendations)
    setPassed(params, "all_user_recommendations")
  }

  // Detect if the parameter was passed; set if so.
  if param.InputModel != nil {
    setCFModel(params, "input_model", param.InputModel)
    setPassed(params, "input_model")
  }

  // Detect if the parameter was passed; set if so.
  if param.Interpolation != "average" {
    setParamString(params, "interpolation", param.Interpolation)
    setPassed(params, "interpolation")
  }

  // Detect if the parameter was passed; set if so.
  if param.IterationOnlyTermination != false {
    setParamBool(params, "iteration_only_termination", param.IterationOnlyTermination)
    setPassed(params, "iteration_only_termination")
  }

  // Detect if the parameter was passed; set if so.
  if param.MaxIterations != 1000 {
    setParamInt(params, "max_iterations", param.MaxIterations)
    setPassed(params, "max_iterations")
  }

  // Detect if the parameter was passed; set if so.
  if param.MinResidue != 1e-05 {
    setParamDouble(params, "min_residue", param.MinResidue)
    setPassed(params, "min_residue")
  }

  // Detect if the parameter was passed; set if so.
  if param.NeighborSearch != "euclidean" {
    setParamString(params, "neighbor_search", param.NeighborSearch)
    setPassed(params, "neighbor_search")
  }

  // Detect if the parameter was passed; set if so.
  if param.Neighborhood != 5 {
    setParamInt(params, "neighborhood", param.Neighborhood)
    setPassed(params, "neighborhood")
  }

  // Detect if the parameter was passed; set if so.
  if param.Normalization != "none" {
    setParamString(params, "normalization", param.Normalization)
    setPassed(params, "normalization")
  }

  // Detect if the parameter was passed; set if so.
  if param.Query != nil {
    gonumToArmaUmat(params, "query", param.Query)
    setPassed(params, "query")
  }

  // Detect if the parameter was passed; set if so.
  if param.Rank != 0 {
    setParamInt(params, "rank", param.Rank)
    setPassed(params, "rank")
  }

  // Detect if the parameter was passed; set if so.
  if param.Recommendations != 5 {
    setParamInt(params, "recommendations", param.Recommendations)
    setPassed(params, "recommendations")
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
  setPassed(params, "output")
  setPassed(params, "output_model")

  // Call the mlpack program.
  C.mlpackCf(params.mem, timers.mem)

  // Initialize result variable and get output.
  var outputPtr mlpackArma
  output := outputPtr.armaToGonumUmat(params, "output")
  var outputModel cfModel
  outputModel.getCFModel(params, "output_model")
  // Clean memory.
  cleanParams(params)
  cleanTimers(timers)
  // Return output(s).
  return output, outputModel
}
