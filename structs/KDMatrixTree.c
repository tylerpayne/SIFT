#include "KDMatrixTree.h"

TreeNode* searchTreeImpl(KDMatrixTree* self, Matrix* feature)
{

}

TreeNode* searchTreeFromImpl(KDMatrixTree* self, TreeNode* n, Matrix* feature)
{

}

void addTreeNodeImpl(KDMatrixTree* self, TreeNode* n)
{

}

void removeTreeNodeImpl(KDMatrixTree* self, TreeNode* n)
{

}

TreeNode* getTreeNodeImpl(KDMatrixTree* self, int idx)
{

}

KDMatrixTree* NewKDMatrixTree(LinkedList* features)
{
  KDMatrixTree* kdtree = (KDMatrixTree*)malloc(sizeof(KDMatrixTree));
  kdtree->get = getTreeNodeImpl;
  kdtree->add = addTreeNodeImpl;
  kdtree->addTrilinearInterpolate = addTrilinearInterpolateImpl;
  kdtree->remove = removeImpl;
  kdtree->search = searchTreeImpl;
  kdtree->searchFromNode = searchTreeFromImpl;

  //Construct Tree

  return kdtree;
}
