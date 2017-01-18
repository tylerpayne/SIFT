#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Array.h"

#define DLLEXPORT extern "C" __declspec (dllexport)
#define CDECL __cdecl

// (i,j) == (row,col)
int IDX2C(int i, int j, int td)
{
  return (i*td)+j;
}

// (retval[0],retval[1]) == (i,j)
int* C2IDX(int i, int td)
{
  int* retval = (int*)malloc(sizeof(int)*2);
  retval[0] = i/td;
  retval[1] = i-(retval[0]*td);
  return retval;
}

typedef struct {
  int x;
  int y;
} Point2;

typedef struct {
  int x;
  int y;
  int z;
} Point3;

typedef struct {
  int x;
  int y;
  int z;
  int w;
} Point4;

typedef struct {
  float x;
  float y;
} Point2f;

typedef struct {
  float x;
  float y;
  float z;
} Point3f;

typedef struct {
  float x;
  float y;
  float z;
  float w;
} Point4f;

typedef struct {
  int width;
  int height;
  Point2 origin;
} Rect;

enum keyType {
  INT = 0,
  FLOAT = 1,
  STRING = 2,
};

enum imType {
  JPEG=0,
  PNG=1,
  ICO=2,
  BMP=3,
};

const char* IMFORMATS[] = {"jpeg","png","ico","bmp"};

typedef struct {
  int ival;
  float fval;
  char* sval;
  int type;
} Key;

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
