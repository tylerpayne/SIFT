#include "Heap.c"

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

PriorityQ* NewPriorityQ(int capacity);
