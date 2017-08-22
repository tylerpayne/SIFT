#include <node.h>

#ifdef __cplusplus
extern "C" {
#endif

int rec_tree_size(Node *root)
{
  if (root->adj_len == 0)
    return 1;
  int size = 0;
  for (int i = 0; i < root->adj_len; i++)
  {
    size += rec_tree_size(root->adjacent[i]);
  }
  return size;
}

int tree_size(Node *root, int *size)
{
  *size = rec_tree_size(root);
  return 0;
}

#ifdef __cplusplus
}
#endif
