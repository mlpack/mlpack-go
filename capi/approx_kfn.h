#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetApproxKFNModelPtr(const char* identifier, void* value);

extern void *mlpackGetApproxKFNModelPtr(const char* identifier);

extern void mlpackApproxKfn();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
