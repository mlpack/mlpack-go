#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetCFModelPtr(const char* identifier, void* value);

extern void *mlpackGetCFModelPtr(const char* identifier);

extern void mlpackCf();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
