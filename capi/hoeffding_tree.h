#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetHoeffdingTreeModelPtr(const char* identifier, void* value);

extern void *mlpackGetHoeffdingTreeModelPtr(const char* identifier);

extern void mlpackHoeffdingTree();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
