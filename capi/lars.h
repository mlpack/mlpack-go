#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetLARSPtr(const char* identifier, void* value);

extern void *mlpackGetLARSPtr(const char* identifier);

extern void mlpackLars();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
