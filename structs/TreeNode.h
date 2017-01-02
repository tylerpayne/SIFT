typedef struct TreeNode TreeNode;

struct TreeNode
{
  Key key;
  void* value;
  TreeNode* leftChild;
  TreeNode* rightChild;
  TreeNode* parent;
};

TreeNode* NewTreeNode(Key key);
