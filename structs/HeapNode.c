#include "TreeNode.h"

int getHeightImpl(TreeNode* self)
{
  return -1;
}

int getDepthImpl(TreeNode* self)
{
  return -1;
}

int getSizeImpl(TreeNode* self)
{
  return -1;
}

int isLeftChildHeapNodeImpl(TreeNode* self)
{
  return -1;
}

int isRightChildHeapNodeImpl(TreeNode* self)
{
  return -1;
}

int isLeafHeapNodeImpl(TreeNode* self)
{
  return -1;
}

int isRootHeapNodeImpl(TreeNode* self)
{
  return -1;
}

int hasLeftChildHeapNodeImpl(TreeNode* self)
{
  return -1;
}

int hasRightChildHeapNodeImpl(TreeNode* self)
{
  return -1;
}
//LESS THAN OR EQUAL
//if 0 then self > node || if 1 then self <= node
int lteToHeapNodeImpl(TreeNode* self, TreeNode* node)
{
  if (self->key.type == STRING)
  {
    return strcmp(self->key.sval,node->key.sval);
  }
  else if (self->key.type == FLOAT)
  {
    return self->key.fval <= node->key.fval;
  } else
  {
    return self->key.ival <= node->key.ival;
  }
  //printf("%i<=%i = %i\n",self->key.ival,node->key.ival,self->key.ival <= node->key.ival);
}

void addRightChildHeapNodeImpl(TreeNode* self, TreeNode* node)
{
  return;
}

void addLeftChildHeapNodeImpl(TreeNode* self, TreeNode* node)
{
  return;
}

void addChildHeapNodeImpl(TreeNode* self, TreeNode* node)
{
  return;
}

void swapHeapNodeImpl(TreeNode* self, TreeNode* node)
{
  return;
}

void freeHeapNodeImpl(TreeNode* self)
{
  free(self->value);
  free(self);
}

TreeNode* NewHeapNode(Key key)
{
  TreeNode* self = (TreeNode*)malloc(sizeof(TreeNode));
  self->key = key;
  self->leftChild = NULL;
  self->rightChild = NULL;
  self->parent = NULL;
  self->value = NULL;

  self->getHeight = getHeightImpl;
  self->getDepth = getDepthImpl;
  self->getSize = getSizeImpl;

  self->isLeftChild = isLeftChildHeapNodeImpl;
  self->isRightChild = isRightChildHeapNodeImpl;
  self->isLeaf = isLeafHeapNodeImpl;
  self->isRoot = isRootHeapNodeImpl;

  self->hasLeftChild = hasLeftChildHeapNodeImpl;
  self->hasRightChild = hasRightChildHeapNodeImpl;
  self->lteTo = lteToHeapNodeImpl;

  self->addRightChild = addRightChildHeapNodeImpl;
  self->addLeftChild = addLeftChildHeapNodeImpl;
  self->add = addChildHeapNodeImpl;
  self->swap = swapHeapNodeImpl;

  self->free = freeHeapNodeImpl;
  return self;
}
