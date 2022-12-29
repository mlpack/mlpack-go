# How to deploy a new mlpack version to mlpack-go.

This process isn't automatic, and the steps below can be refined as time goes on
into an automated script:

 1. Check out mlpack code.
 2. Configure and build the `go` target: `make go`
 3. First, remove the previously auto-generated go-bindings from the root of mlpack-go repository.
```sh
 rm -r -v !(Makefile|Dockerfile|README.md|LICENSE.txt|rel)
```
 4. Then, copy the complete folder of the Go bindings (`build/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack`)
    to the root of mlpack-go repository.
 5. Remove all the `.so` files.
```sh
rm -f *.so *.so.*
```
 6. Manually change the `MLPACK_VERSION` in `Makefile`.
 7. Commit any changed files and any added files in the root and `capi/` folder of mlpack-go repository.
