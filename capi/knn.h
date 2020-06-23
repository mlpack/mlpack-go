#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetKNNModelPtr(const char* identifier, void* value);

extern void *mlpackGetKNNModelPtr(const char* identifier);

extern void mlpackKnn();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
