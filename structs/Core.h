#ifdef EXPORTING
#ifdef __cplusplus
  #define DLLEXPORT extern "C" __declspec (dllexport)
#else
  #define DLLEXPORT __declspec (dllexport)
#endif
#else
#ifdef __cplusplus
  #define DLLEXPORT extern "C" __declspec (dllimport)
#else
  #define DLLEXPORT __declspec (dllimport)
  #endif
#endif

#ifndef _CORE_
#define _CORE_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <structs/Array.h>

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
} Shape;

typedef struct {
  Shape shape;
  Point2 origin;
} Rect;

enum keyType {
  INT = 0,
  FLOAT = 1,
  STRING = 2,
};

typedef enum {
  JPEG=0,
  PNG=1,
  ICO=2,
  BMP=3,
} IMTYPE;

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

int IDX2C(Point2 index, Shape shape)
{
  return (index.y*shape.width)+index.x;
}

Point2 C2IDX(int i, Shape shape)
{
  int y = i/shape.width;
  int x = i-(y*shape.width);
  Point2 retval = {x,y};
  return retval;
}

#endif
