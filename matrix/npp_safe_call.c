#include <core.h>
#include <nppi.h>

#ifdef __cplusplus
extern "C" {
#endif

void npp_safe_call(NppStatus err)
{
  if (err != NPP_SUCCESS)
    printf("NPP Error: %i\n",err);
    //exit(1);
}

#ifdef __cplusplus
}
#endif
