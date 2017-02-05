#include <structs/KDTree.h>

TreeNode* recSearchKDTreeImpl(KDTree* self, TreeNode* node, Matrix* value, int k, PriorityQ* pq)
{
  if (node->isLeaf(node))
  {
    return node;
  }
  if (k == self->kDimensions)
  {
    k = 0;
  }
  Key key = NewFloatKey(value->getElement(value,0,k));
  TreeNode* cmpNode = NewBinaryTreeNode(key);
  int cmp = node->compare(node,cmpNode);
  if (cmp == 0 || cmp == 1)
  {
    if (node->hasLeftChild(node))
    {
      if (node->hasRightChild(node) && pq!=NULL)
      {
        Key otherPathK = NewIntKey(k+1);
        TreeNode* otherPath = NewBinaryTreeNode(otherPathK);
        otherPath->value = (void*)(node->rightChild);
        pq->enQ(pq,otherPath,fabs(1.0/((node->key.fval)-key.fval)));
      }
      return recSearchKDTreeImpl(self,node->leftChild,value,k+1,pq);
    }
    else
    {
      return recSearchKDTreeImpl(self,node->rightChild,value,k+1,pq);
    }

  } else
  {
    if (node->hasRightChild(node))
    {
      if (node->hasLeftChild(node) && pq != NULL)
      {
        Key otherPathK = NewIntKey(k+1);
        TreeNode* otherPath = NewBinaryTreeNode(otherPathK);
        otherPath->value = (void*)(node->leftChild);
        pq->enQ(pq,otherPath,fabs(1.0/((node->key.fval)-key.fval)));
      }
      return recSearchKDTreeImpl(self,node->rightChild,value,k+1,pq);
    }
    else
    {
      return recSearchKDTreeImpl(self,node->leftChild,value,k+1,pq);
    }
  }
}

TreeNode* searchKDTreeImpl(KDTree* self, Matrix* value, TreeNode* knn, int k)
{
  PriorityQ* pq = NewPriorityQ(self->height*2);
  TreeNode* retval = recSearchKDTreeImpl(self,self->root,value,0,pq);
  if (pq->size > 0)
  {
    if (k > 0)
    {
      for (int i = 0; i < k && (pq->size) > 0; i++)
      {
        TreeNode* n = pq->deQ(pq);
        int dim = n->key.ival;
        TreeNode* otherPath = (TreeNode*)(n->value);
        TreeNode* nextBest = recSearchKDTreeImpl(self,otherPath,value,dim,NULL);
        memcpy(knn,nextBest,sizeof(TreeNode));
        knn++;
      }
    }
  }
  free(pq);
  return retval;
}

void recLTPopulateKDTree(KDTree* self, TreeNode* node, TreeNode* nodes, int start, int length, int k);

void recGTPopulateKDTree(KDTree* self, TreeNode* node, TreeNode* nodes, int start, int length, int k)
{
  if (length == 1)
  {
    Keypoint* keypoint = (Keypoint*)(nodes[start].value);
    Matrix* m = (Matrix*)(keypoint->get(keypoint,"feature"));
    Key key = NewFloatKey(m->getElement(m,0,k));
    TreeNode* n = NewBinaryTreeNode(key);
    n->value = (void*)keypoint;
    node->addRightChild(node,n);
    return;
  }
  if (k == self->kDimensions)
  {
    k = 0;
  }
  Heap* heap = NewHeap(length);
  for (int i = start; i < start+length; i++)
  {
    Keypoint* keypoint = (Keypoint*)(nodes[i].value);
    Matrix* m = (Matrix*)(keypoint->get(keypoint,"feature"));
    Key key = NewFloatKey(m->getElement(m,0,k));
    TreeNode* n = NewBinaryTreeNode(key);
    n->value = (void*)keypoint;
    heap->add(heap,n);
  }
  TreeNode* orderedNodes = (TreeNode*)malloc(sizeof(TreeNode)*length);
  for (int i = 0; i < length; i++)
  {
    orderedNodes[i] = *(heap->pop(heap));
  }
  int nMedian = (length/2);
  free(heap);
  node->addRightChild(node,&orderedNodes[nMedian]);
  recGTPopulateKDTree(self,&orderedNodes[nMedian],orderedNodes,0,nMedian,k+1);
  recLTPopulateKDTree(self,&orderedNodes[nMedian],orderedNodes,nMedian,length-(nMedian),k+1);
}

void recLTPopulateKDTree(KDTree* self, TreeNode* node, TreeNode* nodes, int start, int length, int k)
{
  if (length  == 1)
  {
    Keypoint* keypoint = (Keypoint*)(nodes[start].value);
    Matrix* m = (Matrix*)(keypoint->get(keypoint,"feature"));
    Key key = NewFloatKey(m->getElement(m,0,k));
    TreeNode* n = NewBinaryTreeNode(key);
    n->value = (void*)keypoint;
    node->addLeftChild(node,n);
    return;
  }
  if (k == self->kDimensions)
  {
    k = 0;
  }
  Heap* heap = NewHeap(length);
  for (int i = start; i < start+length; i++)
  {
    Keypoint* keypoint = (Keypoint*)(nodes[i].value);
    Matrix* m = (Matrix*)(keypoint->get(keypoint,"feature"));
    Key key = NewFloatKey(m->getElement(m,0,k));
    TreeNode* n = NewBinaryTreeNode(key);
    n->value = (void*)keypoint;
    heap->add(heap,n);
  }
  TreeNode* orderedNodes = (TreeNode*)malloc(sizeof(TreeNode)*length);
  for (int i = 0; i < length; i++)
  {
    orderedNodes[i] = *(heap->pop(heap));
  }
  free(heap);
  int nMedian = (length/2);
  node->addLeftChild(node,&orderedNodes[nMedian]);
  recGTPopulateKDTree(self,&orderedNodes[nMedian],orderedNodes,0,nMedian,k+1);
  recLTPopulateKDTree(self,&orderedNodes[nMedian],orderedNodes,nMedian,length-(nMedian),k+1);
}

void populateKDTree(KDTree* self)
{
  int nPoints = self->nPoints;
  Heap* heap = NewHeap(nPoints);
  for (int i = 0; i < nPoints; i++)
  {
    Keypoint* keypoint = self->data[i];
    Matrix* m = (Matrix*)(keypoint->get(keypoint,"feature"));
    Key key = NewFloatKey(m->getElement(m,0,0));
    TreeNode* n = NewBinaryTreeNode(key);
    n->value = (void*)keypoint;
    heap->add(heap,n);
  }
  TreeNode* orderedNodes = (TreeNode*)malloc(sizeof(TreeNode)*nPoints);
  for (int i = 0; i < nPoints; i++)
  {
    orderedNodes[i] = *(heap->pop(heap));
  }
  int nMedian = (nPoints/2);
  self->root = &orderedNodes[nMedian];
  recGTPopulateKDTree(self,self->root,orderedNodes,0,nMedian,1);
  recLTPopulateKDTree(self,self->root,orderedNodes,nMedian,nPoints-(nMedian),1);
}

DLLEXPORT KDTree* NewKDTree(Keypoint** data, int nPoints, int kDimensions)
{
  printf("Building KDTree\n");
  KDTree* self = (KDTree*)malloc(sizeof(KDTree));
  self->size = 0;
  self->kDimensions = kDimensions;
  self->data = data;
  self->nPoints = nPoints;
  self->search = searchKDTreeImpl;
  populateKDTree(self);
  self->height = self->root->getHeight(self->root);
  return self;
}
