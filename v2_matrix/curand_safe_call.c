#include <core.h>

#ifdef __cplusplus
extern "C" {
#endif

void curand_safe_call(curandStatus_t err)
{
  if (err != CURAND_STATUS_SUCCESS)
    printf("CURAND ERROR: %i\n",err);
    //exit(1);
}

#ifdef __cplusplus
}
#endif
