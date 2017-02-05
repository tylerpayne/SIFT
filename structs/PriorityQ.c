#include <structs/PriorityQ.h>

void enQImpl(PriorityQ* self, TreeNode* node, float priority)
{
  Key k = NewFloatKey(priority);
  TreeNode* n = NewBinaryTreeNode(k);
  n->value = (void*)node;
  self->heap->add(self->heap,n);
  self->size = self->heap->size;
}

TreeNode* deQImpl(PriorityQ* self)
{
  (self->size)--;
  return (TreeNode*)((self->heap->pop(self->heap))->value);
}

DLLEXPORT PriorityQ* NewPriorityQ(int capacity)
{
  PriorityQ* self = (PriorityQ*)malloc(sizeof(PriorityQ));
  self->size = 0;
  self->enQ = enQImpl;
  self->deQ = deQImpl;
  self->heap = NewHeap(capacity);
  return self;
}
