#include "TreeNode.h"

int postorderHeight(TreeNode* self, int height)
{
  int leftSubtree = 0;
  if (self->hasLeftChild(self))
  {
    leftSubtree = postorderHeight(self->leftChild,leftSubtree+1);
  }
  int rightSubtree = 0;
  if (self->hasRightChild(self))
  {
    rightSubtree = postorderHeight(self->rightChild,rightSubtree+1);
  }
  return height + (int)fmax(leftSubtree,rightSubtree);
}

int getHeightImpl(TreeNode* self)
{
  return postorderHeight(self,0);
}

int getDepthImpl(TreeNode* self)
{
  int depth = 0;
  while (!self->isRoot(self))
  {
    self = self->parent;
    depth += 1;
  }
  return depth;
}

int inorderSize(TreeNode* self, int size)
{
  if (self->hasLeftChild(self))
  {
    size = inorderSize(self->leftChild,size);
  }
  size += 1;
  if (self->hasRightChild(self))
  {
    size = inorderSize(self->rightChild,size);
  }
  return size;
}

int getSizeImpl(TreeNode* self)
{
  return inorderSize(self,0);
}

TreeNode* recSearchBinaryTree(TreeNode* self, TreeNode* value)
{
  int cmp = self->compare(self,value);
  if (cmp == 0) //found
  {
    return self;
  }
  else if (cmp == -1) //value is greater than self
  {
    if (self->hasRightChild(self))
    {
      return recSearchBinaryTree(self->rightChild,value);
    } else
    {
      return self;
    }
  } else // value is less than self
  {
    if (self->hasLeftChild(self))
    {
      return recSearchBinaryTree(self->leftChild,value);
    } else
    {
      return self;
    }
  }
}

TreeNode* searchBinaryTreeNodeImpl(TreeNode* self, Key value)
{
  TreeNode* s = NewBinaryTreeNode(value);
  return recSearchBinaryTree(self,s);
}

int isLeftChildBinaryTreeNodeImpl(TreeNode* self)
{
  if (self->parent == NULL)
  {
    return 0;
  }
  return self->parent->leftChild == self;
}

int isRightChildBinaryTreeNodeImpl(TreeNode* self)
{
  if (self->parent == NULL)
  {
    return 0;
  }
  return self->parent->rightChild == self;
}

int isLeafBinaryTreeNodeImpl(TreeNode* self)
{
  if (self->leftChild == NULL && self->rightChild == NULL)
  {
    return 1;
  }
  return 0;
}

int isRootBinaryTreeNodeImpl(TreeNode* self)
{
  if (self->parent == NULL)
  {
    return 1;
  }
  return 0;
}

int hasLeftChildBinaryTreeNodeImpl(TreeNode* self)
{
  if (self->leftChild == NULL)
  {
    return 0;
  }
  return 1;
}

int hasRightChildBinaryTreeNodeImpl(TreeNode* self)
{
  if (self->rightChild == NULL)
  {
    return 0;
  }
  return 1;
}
//LESS THAN OR EQUAL
//if 0 then node > self || if 1 then node <= self
int lteToBinaryTreeNodeImpl(TreeNode* self, TreeNode* node)
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
}

//-1 = self<node; 0 = self==node; 1 = self>node;
int compareBinaryTreeNodeImpl(TreeNode* self, TreeNode* node)
{
  if (self->key.type == STRING)
  {
    //strcmp
  }
  if (self->key.type == FLOAT)
  {
    if (self->key.fval < node->key.fval)
    {
      return -1;
    } else if (self->key.fval == node->key.fval)
    {
      return 0;
    } else
    {
      return 1;
    }
  }
  if (self->key.ival < node->key.ival)
  {
    return -1;
  } else if (self->key.ival == node->key.ival)
  {
    return 0;
  } else
  {
    return 1;
  }
}

void addRightChildBinaryTreeNodeImpl(TreeNode* self, TreeNode* node)
{
  if (self->hasRightChild(self))
  {
    if (self->rightChild->lteTo(self->rightChild,node))
    {
      node->leftChild = self->rightChild;
      self->rightChild->parent = node;
      node->parent = self;
      self->rightChild = node;
    }
    else
    {
      node->rightChild = self->rightChild;
      self->rightChild->parent = node;
      node->parent = self;
      self->rightChild = node;
    }
  }
  else
  {
    self->rightChild = node;
    node->parent = self;
  }

}

void addLeftChildBinaryTreeNodeImpl(TreeNode* self, TreeNode* node)
{
  if (self->hasLeftChild(self))
  {
    if (self->leftChild->lteTo(self->leftChild,node))
    {
      node->leftChild = self->leftChild;
      self->leftChild->parent = node;
      node->parent = self;
      self->leftChild = node;
    }
    else
    {
      node->rightChild = self->leftChild;
      self->leftChild->parent = node;
      node->parent = self;
      self->leftChild = node;
    }
  } else
  {
    self->leftChild = node;
    node->parent = self;
  }

}

void addChildBinaryTreeNodeImpl(TreeNode* self, TreeNode* node)
{
  TreeNode* addNode = self->search(self,node->key);
  if (addNode->lteTo(addNode,node))
  {
    addNode->addRightChild(addNode,node);
  } else //addNode > node
  {
    addNode->addLeftChild(addNode,node);
  }
}

void swapBinaryTreeNodeImpl(TreeNode* self, TreeNode* node)
{
  TreeNode* lc = self->leftChild;
  TreeNode* rc = self->rightChild;
  TreeNode* p = self->parent;
  self->leftChild = node->leftChild;
  self->rightChild = node->rightChild;
  self->parent = node->parent;
  node->leftChild = lc;
  node->rightChild = rc;
  node->parent = p;
}

void freeBinaryTreeNodeImpl(TreeNode* self)
{
  free(self->value);
  free(self);
}

TreeNode* NewBinaryTreeNode(Key key)
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

  self->isLeftChild = isLeftChildBinaryTreeNodeImpl;
  self->isRightChild = isRightChildBinaryTreeNodeImpl;
  self->isLeaf = isLeafBinaryTreeNodeImpl;
  self->isRoot = isRootBinaryTreeNodeImpl;

  self->hasLeftChild = hasLeftChildBinaryTreeNodeImpl;
  self->hasRightChild = hasRightChildBinaryTreeNodeImpl;
  self->lteTo = lteToBinaryTreeNodeImpl;
  self->compare = compareBinaryTreeNodeImpl;

  self->addRightChild = addRightChildBinaryTreeNodeImpl;
  self->addLeftChild = addLeftChildBinaryTreeNodeImpl;
  self->add = addChildBinaryTreeNodeImpl;
  self->swap = swapBinaryTreeNodeImpl;

  self->search = searchBinaryTreeNodeImpl;

  self->free = freeBinaryTreeNodeImpl;
  return self;
}
