#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetLinearSVMModelPtr(const char* identifier, void* value);

extern void *mlpackGetLinearSVMModelPtr(const char* identifier);

extern void mlpackLinearSvm();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
