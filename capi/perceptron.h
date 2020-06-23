#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern void mlpackSetPerceptronModelPtr(const char* identifier, void* value);

extern void *mlpackGetPerceptronModelPtr(const char* identifier);

extern void mlpackPerceptron();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
