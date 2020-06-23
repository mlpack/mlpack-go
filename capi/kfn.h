#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetKFNModelPtr(const char* identifier, void* value);

extern void *mlpackGetKFNModelPtr(const char* identifier);

extern void mlpackKfn();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
