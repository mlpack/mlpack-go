/**
 * @file capi/sparse_coding.h
 *
 * This is an autogenerated header file for functions specified to the %NAME%
 * binding to be called by Go.
 */
#ifndef GO_sparse_coding_H
#define GO_sparse_coding_H

#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)

extern "C"
{
#endif

extern void mlpackSparseCoding(void* params, void* timers);

// Any definitions of methods for dealing with model pointers will be put below
// this comment, if needed.

// Set the pointer to a SparseCoding<> parameter.
extern void mlpackSetSparseCodingPtr(void* params,
                                           const char* identifier,
                                           void* value);

// Get the pointer to a SparseCoding<> parameter.
extern void* mlpackGetSparseCodingPtr(void* params,
                                            const char* identifier);


#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
