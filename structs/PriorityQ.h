#include <stdlib.h>
#include "LinkedList.c"

typedef struct PriorityQ PriorityQ;

typedef void (*enqFunc)(PriorityQ* self, ListNode *, float);
typedef ListNode* (*deqFunc)(PriorityQ* self);

struct PriorityQ
{
  LinkedList* list;
  enqFunc enQ;
  deqFunc deQ;
};

PriorityQ* NewPriorityQ();
