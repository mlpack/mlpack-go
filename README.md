<h2 align="center">
  <a href="http://mlpack.org"><img
src="https://cdn.rawgit.com/mlpack/mlpack.org/e7d36ed8/mlpack-black.svg"
style="background-color:rgba(0,0,0,0);" height=230 alt="mlpack: a fast, flexible
machine learning library"></a>
  <br>a fast, flexible machine learning library<br>
</h2>

<h5 align="center">
  <a href="https://mlpack.org">Home</a> |
  <a href="https://www.mlpack.org/docs.html">Documentation</a> |
  <a href="https://www.mlpack.org/doc/mlpack-git/doxygen/index.html">Doxygen</a>
|
  <a href="https://www.mlpack.org/community.html">Community</a> |
  <a href="https://www.mlpack.org/questions.html">Help</a> |
  <a href="https://webchat.freenode.net/?channels=mlpack">IRC Chat</a>
</h5>

This repository contains the *Go bindings* for mlpack.  These bindings are
auto-generated, and so this repository is not really maintained or monitored.

If you are *looking for the documentation* for the Go bindings, try here:

 * [Go binding documentation](https://www.mlpack.org/doc/mlpack-git/go_documentation.html)

If you are *having trouble* or *want to learn more*, try looking in the main
mlpack repository:

 * [mlpack/mlpack](https://github.com/mlpack/mlpack/)

Any issues with the Go bindings should be filed there.

## How to use

## Simple mlpack quickstart example

As a really simple example of how to use mlpack from Go, let's do some
simple classification on a subset of the standard machine learning  `covertype`
dataset.  We'll first split the dataset into a training set and a testing set,
then we'll train an mlpack random forest on the training data, and finally we'll
print the accuracy of the random forest on the test dataset.

```go
package main

import (
  "mlpack.org/v1/mlpack"
  "fmt"
)
func main() {

  // Download dataset.
  mlpack.DownloadFile("https://www.mlpack.org/datasets/covertype-small.data.csv.gz",
                      "data.csv.gz")
  mlpack.DownloadFile("https://www.mlpack.org/datasets/covertype-small.labels.csv.gz",
                      "labels.csv.gz")

  // Extract/Unzip the dataset.
  mlpack.UnZip("data.csv.gz", "data.csv")
  dataset, _ := mlpack.Load("data.csv")

  mlpack.UnZip("labels.csv.gz", "labels.csv")
  labels, _ := mlpack.Load("labels.csv")

  // Split the dataset using mlpack.
  params := mlpack.PreprocessSplitOptions()
  params.InputLabels = labels
  params.TestRatio = 0.3
  params.Verbose = true
  test, test_labels, train, train_labels :=
      mlpack.PreprocessSplit(dataset, params)

  // Train a random forest.
  rf_params := mlpack.RandomForestOptions()
  rf_params.NumTrees = 10
  rf_params.MinimumLeafSize = 3
  rf_params.PrintTrainingAccuracy = true
  rf_params.Training = train
  rf_params.Labels = train_labels
  rf_params.Verbose = true
  rf_model, _, _ := mlpack.RandomForest(rf_params)

  // Predict the labels of the test points.
  rf_params_2 := mlpack.RandomForestOptions()
  rf_params_2.Test = test
  rf_params_2.InputModel = &rf_model
  rf_params_2.Verbose = true
  _, predictions, _ := mlpack.RandomForest(rf_params_2)

  // Now print the accuracy.
  rows, _ := predictions.Dims()
  var sum int = 0
  for i := 0; i < rows; i++ {
    if (predictions.At(i, 0) == test_labels.At(i, 0)) {
      sum = sum + 1
    }
  }
  fmt.Print(sum, " correct out of ", rows, " (",
      (float64(sum) / float64(rows)) * 100, "%).\n")
}
```
We can see that we achieve reasonably good accuracy on the test dataset (80%+); if we use the full `covertype.csv.gz`, the accuracy should increase significantly (but training will take longer).
It's easy to modify the code above to do more complex things, or to use different mlpack learners, or to interface with other machine learning toolkits.


## Using mlpack for movie recommendations

In this example, we'll train a collaborative filtering model using mlpack's
<tt><a href="https://godoc.org/mlpack.org/v1/mlpack#Cf">Cf()</a></tt> method.  We'll train this on the MovieLens dataset from
https://grouplens.org/datasets/movielens/, and then we'll use the model that we
train to give recommendations.

```go
package main

import (
  "github.com/frictionlessdata/tableschema-go/csv"
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
  "fmt"
)
func main() {

  // Download dataset.
  mlpack.DownloadFile("https://www.mlpack.org/datasets/ml-20m/ratings-only.csv.gz",
                      "ratings-only.csv.gz")
  mlpack.DownloadFile("https://www.mlpack.org/datasets/ml-20m/movies.csv.gz",
                      "movies.csv.gz")

  // Extract dataset.
  mlpack.UnZip("ratings-only.csv.gz", "ratings-only.csv")
  ratings, _ := mlpack.Load("ratings-only.csv")

  mlpack.UnZip("movies.csv.gz", "movies.csv")
  table, _ := csv.NewTable(csv.FromFile("movies.csv"), csv.LoadHeaders())
  movies, _ := table.ReadColumn("title")

  // Split the dataset using mlpack.
  params := mlpack.PreprocessSplitOptions()
  params.TestRatio = 0.1
  params.Verbose = true
  ratings_test, _, ratings_train, _ := mlpack.PreprocessSplit(ratings, params)

  // Train the model.  Change the rank to increase/decrease the complexity of the
  // model.
  cf_params := mlpack.CfOptions()
  cf_params.Training = ratings_train
  cf_params.Test = ratings_test
  cf_params.Rank = 10
  cf_params.Verbose = true
  cf_params.Algorithm = "RegSVD"
  _, cf_model := mlpack.Cf(cf_params)

  // Now query the 5 top movies for user 1.
  cf_params_2 := mlpack.CfOptions()
  cf_params_2.InputModel = &cf_model
  cf_params_2.Recommendations = 10
  cf_params_2.Query = mat.NewDense(1, 1, []float64{1})
  cf_params_2.Verbose = true
  cf_params_2.MaxIterations = 10
  output, _ := mlpack.Cf(cf_params_2)

  // Get the names of the movies for user 1.
  fmt.Println("Recommendations for user 1")
  for i := 0; i < 10; i++ {
    fmt.Println(i, ":", movies[int(output.At(0 , i))])
  }
}
```

Here is some example output, showing that user 1 seems to have good taste in movies:

	Recommendations for user 1:
	Casablanca (1942)
	Pan's Labyrinth (Laberinto del fauno, El) (2006)
	Godfather, The (1972)
	Answer This! (2010)
	Life Is Beautiful (La Vita è bella) (1997)
	Adventures of Tintin, The (2011)
	Dark Knight, The (2008)
	Out for Justice (1991)
	Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
	Schindler's List (1993)


## How to install

To install mlpack, run the following command:

```
go get -u -d mlpack.org/v1/mlpack
```

To run code that uses the mlpack package, you must also install mlpack 3.3.2 and go shared libraries on your system. Here are instructions.

## Ubuntu/Linux

### Installation

You can use `make` to install mlpack 3.3.2 and other go-shared libraries with the handy `Makefile` included with this repo. The installation performed by the `Makefile` is minimal, so it may remove mlpack options such as Python or Julia bindings if you have already installed mlpack some other way.

#### Quick Install

The following commands should do everything to download and mlpack 3.3.2 on Linux:

	cd $GOPATH/src/mlpack.org/org/mlpack
	make install

If it works correctly, at the end of the entire process, the following message should be displayed:

	PASS
	ok	mlpack.org/v1/mlpack/tests

That's it, now you are ready to use mlpack.

#### Complete Install

If you have already done the "Quick Install" as described above, you do not need to run any further commands. For the curious, or for custom installations, here are the details for each of the steps that are performed when you run `make install`.

##### Install required packages

First, you need to change the current directory to the location of the mlpack repo, so you can access the `Makefile`:

	cd $GOPATH/src/mlpack.org/v1/mlpack

Next, you need to update the system, and install any required packages:

	make deps

#### Download source

Now, download the mlpack 3.3.2 source code:

	make download

#### Build

Build everything. This will take quite a while:

	make build

#### Install

Once the code is built, you are ready to install:

	make sudo_install

### Verifying the installation

To verify your installation you can run tests.

First, change the current directory to the location of the mlpack repo:

	cd $GOPATH/src/mlpack.org/v1/mlpack

Now you should be able to build or run any of the examples:

	go test ./tests/

The version program should output the following:

	ok	mlpack.org/v1/mlpack/tests

#### Cleanup extra files

After the installation is complete, you can remove the extra files and folders:

	make clean
