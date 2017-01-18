#include "Heap.h"

int nParentOf(int idx)
{
  return (idx-1)/2;
}

int nLeftChildOf(int idx)
{
  return 2*idx + 1;
}

int nRightChildOf(int idx)
{
  return 2*idx + 2;
}

TreeNode* parentOf(Heap* self, int idx)
{
  return &(self->nodes[nParentOf(idx)]);
}

TreeNode* leftChildOf(Heap* self, int idx)
{
  return &(self->nodes[nLeftChildOf(idx)]);
}

TreeNode* rightChildOf(Heap* self, int idx)
{
  return &(self->nodes[nRightChildOf(idx)]);
}

void addHeapImpl(Heap* self, TreeNode* node)
{
  if (self->size < self->maxSize)
  {
    self->nodes[self->size] = *node;
    (self->size)++;
    if (self->size > 0)
    {
      self->reheapUp(self);
    }
  }
  else
  {
    TreeNode* newNodes = (TreeNode*)malloc(sizeof(TreeNode)*self->size*2);
    memcpy(newNodes,self->nodes,sizeof(TreeNode)*self->size);
    free(self->nodes);
    self->nodes = newNodes;
    self->maxSize = self->size*2;
    self->add(self,node);
  }
}

void removeHeapImpl(Heap* self)
{
  (self->size)--;
}

TreeNode* popHeapImpl(Heap* self)
{
  TreeNode* retval = (TreeNode*)malloc(sizeof(TreeNode));
  memcpy(retval,self->nodes,sizeof(TreeNode));
  self->swap(self,0,self->size-1);
  self->remove(self);
  self->reheapDown(self);
  return retval;
}

void heapifyImpl(Heap* self, Array treenodeArray)
{
  for (int i = 0; i < treenodeArray.count; i++)
  {
    self->add(self,&(((TreeNode*)treenodeArray.ptr)[i]));
  }
}

void swapHeapImpl(Heap* self, int n, int u)
{
  TreeNode tmp1 = self->nodes[n];
  TreeNode tmp2 = self->nodes[u];
  self->nodes[n] = tmp2;
  self->nodes[u] = tmp1;
}

void reheapUpImpl(Heap* self)
{
  int idx = self->size-1;
  TreeNode* parent = parentOf(self, idx);
  TreeNode* node = &(self->nodes[idx]);
  while (parent->lteTo(parent,node))
  {
    if (nParentOf(idx) == idx)
    {
      break;
    }
    self->swap(self,idx,nParentOf(idx));
    idx = nParentOf(idx);
    parent = parentOf(self,idx);
    node = &(self->nodes[idx]);
  }
}

void reheapDownImpl(Heap* self)
{
  int idx = 0;
  int cond = 1;
  while (cond==1)
  {
    TreeNode* parent = &(self->nodes[idx]);
    TreeNode* leftChild = leftChildOf(self,idx);
    TreeNode* rightChild = rightChildOf(self,idx);
    if (idx >= (self->size)-1)
    {
      cond = 0;
      break;
    }
    if (nLeftChildOf(idx) >= (self->size))
    {
      cond = 0;
      break;
    }
    else
    {
      if (nRightChildOf(idx) < (self->size))
      {
        if (leftChild->lteTo(leftChild,rightChild))
        {
          if (parent->lteTo(parent,rightChild))
          {
            self->swap(self,idx,nRightChildOf(idx));
            idx = nRightChildOf(idx);
          } else
          {
            cond = 0;
          }

        }
        else
        {
          if (parent->lteTo(parent,leftChild))
          {
            self->swap(self,idx,nLeftChildOf(idx));
            idx = nLeftChildOf(idx);
          } else
          {
            cond = 0;
          }
        }
      }
      else
      {
        //no right child, only left child
        if (parent->lteTo(parent,leftChild))
        {
          self->swap(self,idx,nLeftChildOf(idx));
          idx = nLeftChildOf(idx);
        } else
        {
          cond = 0;
        }
      }
    }
  }
}

Heap* NewHeap(int maxSize)
{
  Heap* self = (Heap*)malloc(sizeof(Heap));
  self->nodes = (TreeNode*)malloc(sizeof(TreeNode)*maxSize);
  self->size = 0;
  self->maxSize = maxSize;

  self->add = addHeapImpl;
  self->pop = popHeapImpl;
  self->remove = removeHeapImpl;
  self->heapify = heapifyImpl;
  self->reheapUp = reheapUpImpl;
  self->reheapDown = reheapDownImpl;
  self->swap = swapHeapImpl;
  return self;

}
