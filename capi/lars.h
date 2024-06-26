/**
 * @file capi/lars.h
 *
 * This is an autogenerated header file for functions specified to the %NAME%
 * binding to be called by Go.
 */
#ifndef GO_lars_H
#define GO_lars_H

#include <stddef.h>

#if defined(__cplusplus) || defined(c_plusplus)

extern "C"
{
#endif

extern void mlpackLars(void* params, void* timers);

// Any definitions of methods for dealing with model pointers will be put below
// this comment, if needed.

// Set the pointer to a LARS<> parameter.
extern void mlpackSetLARSPtr(void* params,
                                           const char* identifier,
                                           void* value);

// Get the pointer to a LARS<> parameter.
extern void* mlpackGetLARSPtr(void* params,
                                            const char* identifier);


#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
