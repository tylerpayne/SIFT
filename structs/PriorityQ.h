#include "Heap.c"

typedef struct PriorityQ PriorityQ;

typedef void (*enqFunc)(PriorityQ* self, ListNode *, float);
typedef ListNode* (*deqFunc)(PriorityQ* self);

struct PriorityQ
{
  Heap* heap;
  enqFunc enQ;
  deqFunc deQ;
};

PriorityQ* NewPriorityQ();
