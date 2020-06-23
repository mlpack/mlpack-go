#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetLinearRegressionPtr(const char* identifier, void* value);

extern void *mlpackGetLinearRegressionPtr(const char* identifier);

extern void mlpackLinearRegression();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
