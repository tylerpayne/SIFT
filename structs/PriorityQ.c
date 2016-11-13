#include "LinkedList.h"

void enQImpl(PriorityQ* self, ListNode* node, float priority)
{
  node->key = (void*) &priority;
  if (self->list->firstNode == null)
  {
    self->list->append(self->list,node);
  } else
  {
    ListNode* u = self->list->firstNode;
    int i = 0;
    float ukey = *(float*)u->key;
    while (ukey < priority && u->nextNode != NULL)
    {
      u = u->nextNode;
      ukey = *(float*)u->key;
      i++;
    }
    u->list->insert(u->list,node,idx);
  }
}

ListNode* deQImpl(PriorityQ* self)
{
  if (self->list->lastNode == NULL)
  {
    return NULL;
  }
  else
  {
    ListNode* retval = self->list->pop(self->list,self->list->count-1);
    return retval;
  }
}

PriorityQ* NewPriorityQ()
{
  PriorityQ* pq = (PriorityQ*)malloc(sizeof(PriorityQ));
  pq.enQ = enQImpl;
  pq.deQ = deQImpl;
  return pq;
}
