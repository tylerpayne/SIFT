#include "TreeNode.c"

typedef struct BinarySearchTree BinarySearchTree;

typedef void (*addBSTNodeFunc)(BinarySearchTree*,TreeNode*);
typedef TreeNode* (*searchBSTKeyFunc)(BinarySearchTree*,float);

struct BinarySearchTree
{
  int size;
  TreeNode* root;
  addBSTNodeFunc add;
  searchBSTKeyFunc search;
  addBSTNodeFunc remove;
};

BinarySearchTree* NewBinarySearchTree();
