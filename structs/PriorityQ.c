#include "PriorityQ.h"

void enQImpl(PriorityQ* self, ListNode* node, float priority)
{
  node->key.fval = priority;
  if (self->list->count == 0)
  {
    self->list->append(self->list,node);
  } else
  {
    ListNode* u = self->list->firstNode;
    int i = 0;
    while (u->key.fval < priority)
    {
      if (u->nextNode == NULL)
      {
        i++;
        break;
      }
      u = u->nextNode;
      i++;
    }
    self->list->insert(self->list,node,i);
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
  pq->enQ = enQImpl;
  pq->deQ = deQImpl;
  pq->list = NewLinkedList();
  return pq;
}
