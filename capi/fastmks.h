#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetFastMKSModelPtr(const char* identifier, void* value);

extern void *mlpackGetFastMKSModelPtr(const char* identifier);

extern void mlpackFastmks();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
