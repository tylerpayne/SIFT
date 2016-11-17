#include <stdlib>

typedef struct TreeNode TreeNode;

struct TreeNode
{
  int idx;
  float key;
  void* value;
  TreeNode* leftChild;
  TreeNode* rightChild;
  TreeNode* parent;
};

TreeNode* NewTreeNode(float key);
