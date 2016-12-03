#include "Keypoint.h"

void setKeyValKeypointImpl(Keypoint* self, char* key, void* val)
{
  ListNode* node = NewListNode(val);
  node->key.sval = key;
  self->dictionary->append(self->dictionary,node);
}

void* getKeyKeypointImpl(Keypoint* self, char* key)
{
  return self->dictionary->getKey(self->dictionary,key)->value;
}

void removeKeyKeypointImpl(Keypoint* self, char* key)
{
  self->dictionary->remove(self->dictionary,self->dictionary->getKey(self->dictionary,key)->key.ival);
}

Keypoint* NewKeypoint(float x, float y, Image* image)
{
  Keypoint* kp = (Keypoint*)malloc(sizeof(Keypoint));
  List* list = NewList();

  float* pos = (float*)malloc(sizeof(float)*2);
  pos[0] = x;
  pos[1] = y;

  kp->dictionary = list;
  kp->position = pos;
  kp->set = setKeyValKeypointImpl;
  kp->get = getKeyKeypointImpl;
  kp->remove = removeKeyKeypointImpl;

  return kp;
}
