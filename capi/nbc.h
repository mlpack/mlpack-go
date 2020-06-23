#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetNBCModelPtr(const char* identifier, void* value);

extern void *mlpackGetNBCModelPtr(const char* identifier);

extern void mlpackNbc();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
