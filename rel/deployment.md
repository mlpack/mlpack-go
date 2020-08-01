# How to deploy a new mlpack version to mlpack-go.

This process isn't automatic, and the steps below can be refined as time goes on
into an automated script:

 1. Check out mlpack code.
 2. Configure and build the `go` target: `make go`
 3. Copy the complete folder of the Go bindings (`build/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack`)
    to the root of mlpack-go repository.
 4. Remove all the `.so` files.
```sh
rm -f *.so *.so.*
```
 5. Manually change the `PACKAGE_VERSION` in `MAKEFILE`.
 6. Commit any changed files and any added files in `root` and `capi/` folder of mlpack-go repository.
