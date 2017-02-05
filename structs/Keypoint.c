#include <structs/Keypoint.h>

void setKeyValKeypointImpl(Keypoint* self, char* key, void* val)
{
  ListNode* node = NewListNode(val);
  node->key = NewStringKey(key);
  self->dictionary->append(self->dictionary,node);
}

void* getKeyKeypointImpl(Keypoint* self, char* key)
{
  void* retval = (self->dictionary->getKey(self->dictionary,key));
  if (retval == NULL)
  {
    if (VERBOSITY > 2)
    {
      printf("FAILED TO GET VALUE FOR KEY: %s\n",key);
    }
    return NULL;
  }
  return ((ListNode*)retval)->value;
}

void removeKeyKeypointImpl(Keypoint* self, char* key)
{
  self->dictionary->remove(self->dictionary,self->dictionary->getKey(self->dictionary,key)->key.ival);
}

DLLEXPORT Keypoint* NewKeypoint(float x, float y, Image* image)
{
  Keypoint* kp = (Keypoint*)malloc(sizeof(Keypoint));
  List* list = NewList();

  Point2f pos = {x,y};

  kp->dictionary = list;
  kp->position = pos;
  kp->set = setKeyValKeypointImpl;
  kp->get = getKeyKeypointImpl;
  kp->remove = removeKeyKeypointImpl;

  if (image != NULL)
  {
    kp->sourceImage = image;
  }

  return kp;
}
