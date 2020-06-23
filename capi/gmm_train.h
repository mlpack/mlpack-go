#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetGMMPtr(const char* identifier, void* value);

extern void *mlpackGetGMMPtr(const char* identifier);

extern void mlpackGmmTrain();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
