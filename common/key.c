#include <core.h>

#ifdef __cplusplus
extern "C" {
#endif

Key NewIntKey(int i)
{
  Key k;
  k.ival = i;
  k.type = INT;
  return k;
}

Key NewFloatKey(float i)
{
  Key k;
  k.fval = i;
  k.type = FLOAT;
  return k;
}

Key NewStringKey(char* s)
{
  Key k;
  k.sval = s;
  k.type = STRING;
  return k;
}

#ifdef __cplusplus
}
#endif
