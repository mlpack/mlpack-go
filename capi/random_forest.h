#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetRandomForestModelPtr(const char* identifier, void* value);

extern void *mlpackGetRandomForestModelPtr(const char* identifier);

extern void mlpackRandomForest();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
