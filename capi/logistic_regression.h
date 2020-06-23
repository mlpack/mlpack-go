#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetLogisticRegressionPtr(const char* identifier, void* value);

extern void *mlpackGetLogisticRegressionPtr(const char* identifier);

extern void mlpackLogisticRegression();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
