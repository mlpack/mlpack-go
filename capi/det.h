#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetDTreePtr(const char* identifier, void* value);

extern void *mlpackGetDTreePtr(const char* identifier);

extern void mlpackDet();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
