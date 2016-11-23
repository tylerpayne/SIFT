#include "BinarySearchTree.h"

int isRightChildBSTImpl(TreeNode* this)
{
  if (this == this->parent->rightChild)
  {
    return 1;
  } else
  {
    return 0;
  }
}

int isLeftChildBSTImpl(TreeNode* this)
{
  if (this == this->parent->leftChild)
  {
    return 1;
  }
  else {
      return 0;
  }
}

TreeNode* predBSTImpl(TreeNode* this)
{
  if (this == NULL)
  {
    return NULL;
  }
  if (this->leftChild != NULL)
  {
    TreeNode* u = this->leftChild;
    while (u->rightChild != NULL)
    {
        u = u->rightChild;
    }
    return u;
  } else if (this->parent !=NULL)
  {
    if (isRightChildBSTImpl(this))
    {
      return this->parent;
    } else
    {
      return NULL;
    }
  }
}

TreeNode* succBSTImpl(TreeNode* this)
{
  if (this == NULL)
  {
    return NULL;
  }
  if (this->rightChild != NULL)
  {
    TreeNode* u = this->rightChild;
    while (u->leftChild != NULL)
    {
        u = u->leftChild;
    }
    return u;
  } else if (this->parent !=NULL)
  {
    if (isLeftChildBSTImpl(this))
    {
      return this->parent;
    } else
    {
      return NULL;
    }
  }
}

TreeNode* recSearchBSTImpl(TreeNode* this, float key)
{
  if (this == NULL)
  {
    return NULL;
  } else if (key < this->key)
  {
    return recSearchBSTImpl(this->leftChild,key);
  } else if (key > this->key)
  {
    return recSearchBSTImpl(this->rightChild,key);
  } else
  {
    return this;
  }
}

TreeNode* searchBSTImpl(BinarySearchTree* self, float key)
{
  return recSearchBSTImpl(self->root,key);
}

TreeNode* recFindAddPositionBSTImpl(TreeNode* this, TreeNode* add)
{
  if (add->key < this->key)
  {
    if (this->leftChild == NULL)
    {
      return this;
    } else
    {
      return recFindAddPositionBSTImpl(this->leftChild,add);
    }
  } else if (add->key > this->key)
  {
    if (this->rightChild == NULL)
    {
      return this;
    } else
    {
      return recFindAddPositionBSTImpl(this->rightChild,add);
    }
  } else
  {
    return this;
  }
}

void addBSTNodeImpl(BinarySearchTree* self, TreeNode* node)
{

  if (self->root == NULL)
  {
    self->root = node;
    self->size++;
    return;
  }

  TreeNode* addNode = recFindAddPositionBSTImpl(self->root,node);
  if (node->key <= addNode->key)
  {
    if (addNode->leftChild == NULL)
    {
      addNode->leftChild = node;
      node->parent = addNode;
    } else
    {
      node->parent = addNode;
      node->leftChild = addNode->leftChild;
      addNode->leftChild->parent = node;
      addNode->leftChild = node;
    }

  } else
  {
    if (addNode->rightChild == NULL)
    {
      addNode->rightChild = node;
      node->parent = addNode;
    } else
    {
      node->parent = addNode;
      node->rightChild = addNode->rightChild;
      addNode->rightChild->parent = node;
      addNode->rightChild = node;
    }
  }
  self->size++;
}


BinarySearchTree* NewBinarySearchTree()
{
  BinarySearchTree* bst = (BinarySearchTree*)malloc(sizeof(BinarySearchTree));
  bst->size = 0;
  bst->root = NULL;
  bst->search = searchBSTImpl;
  bst->add = addBSTNodeImpl;
  return bst;
}
