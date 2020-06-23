#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetDSModelPtr(const char* identifier, void* value);

extern void *mlpackGetDSModelPtr(const char* identifier);

extern void mlpackDecisionStump();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
