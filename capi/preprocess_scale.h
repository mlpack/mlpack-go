#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetScalingModelPtr(const char* identifier, void* value);

extern void *mlpackGetScalingModelPtr(const char* identifier);

extern void mlpackPreprocessScale();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
