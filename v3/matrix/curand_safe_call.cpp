#include <core.h>

void curand_safe_call(curandStatus_t err)
{
  if (err != CURAND_STATUS_SUCCESS)
    printf("CURAND ERROR: %i\n",err);
    //exit(1);
}
