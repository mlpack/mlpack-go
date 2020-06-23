#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetSoftmaxRegressionPtr(const char* identifier, void* value);

extern void *mlpackGetSoftmaxRegressionPtr(const char* identifier);

extern void mlpackSoftmaxRegression();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
