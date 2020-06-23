#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetLSHSearchPtr(const char* identifier, void* value);

extern void *mlpackGetLSHSearchPtr(const char* identifier);

extern void mlpackLsh();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
