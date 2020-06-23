#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetDecisionTreeModelPtr(const char* identifier, void* value);

extern void *mlpackGetDecisionTreeModelPtr(const char* identifier);

extern void mlpackDecisionTree();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
