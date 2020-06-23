#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetAdaBoostModelPtr(const char* identifier, void* value);

extern void *mlpackGetAdaBoostModelPtr(const char* identifier);

extern void mlpackAdaboost();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
