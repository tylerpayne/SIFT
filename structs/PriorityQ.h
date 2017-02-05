#include <structs/Heap.h>

#ifndef _PRIORITYQ_
#define _PRIORITYQ_

typedef struct PriorityQ PriorityQ;

typedef void (*enqFunc)(PriorityQ* self, TreeNode *, float);
typedef TreeNode* (*deqFunc)(PriorityQ* self);

struct PriorityQ
{
  int size;
  Heap* heap;
  enqFunc enQ;
  deqFunc deQ;
};

DLLEXPORT PriorityQ* NewPriorityQ(int capacity);
#endif
