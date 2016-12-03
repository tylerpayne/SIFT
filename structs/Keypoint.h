#include <stdlib.h>
#include "List.c"

typedef struct Keypoint Keypoint;

typedef void (*addKeyValKeypointFunc)(Keypoint*,char*,void*);
typedef void (*removeKeyKeypointFunc)(Keypoint*, char*);
typedef void* (*getKeyKeypointFunc)(Keypoint*, char*);

struct Keypoint
{
  Image* sourceImage;
  float* position;
  List* dictionary;
  addKeyValKeypointFunc set;
  getKeyKeypointFunc get;
  removeKeyKeypointFunc remove;
};

Keypoint* NewKeypoint(float x, float y, Image* image);
