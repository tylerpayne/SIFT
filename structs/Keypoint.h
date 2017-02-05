#include <structs/Matrix.h>
#include <structs/Image.h>
#include <structs/ListNode.h>
#include <structs/List.h>

#ifndef _KEYPOINT_
#define _KEYPOINT_

typedef struct Keypoint Keypoint;

typedef void (*addKeyValKeypointFunc)(Keypoint*,char*,void*);
typedef void (*removeKeyKeypointFunc)(Keypoint*, char*);
typedef void* (*getKeyKeypointFunc)(Keypoint*, char*);

struct Keypoint
{
  Image* sourceImage;
  Point2f position;
  List* dictionary;
  addKeyValKeypointFunc set;
  getKeyKeypointFunc get;
  removeKeyKeypointFunc remove;
};

DLLEXPORT Keypoint* NewKeypoint(float x, float y, Image* image);
#endif _KEYPOINT_
