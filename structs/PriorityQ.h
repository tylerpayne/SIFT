#include <stdlib.h>
#include "LinkedList.h"

typedef void (*enqFunc)(PriorityQ* self, ListNode *, float);
typedef ListNode* (PriorityQ* self, *deqFunc)();

typedef struct sPriorityQ
{
  LinkedList* list;
  enqFunc enQ;
  deqFunc deQ;
} PriorityQ;

PriorityQ* NewPriorityQ();
