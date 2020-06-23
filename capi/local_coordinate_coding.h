#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetLocalCoordinateCodingPtr(const char* identifier, void* value);

extern void *mlpackGetLocalCoordinateCodingPtr(const char* identifier);

extern void mlpackLocalCoordinateCoding();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
