#include "List.h"

ListNode* getImpl(List* self, int idx)
{
  if (idx < self->count)
  {
    int i = 0;
    ListNode* u = self->firstNode;
    if (u->nextNode != NULL)
    {
      while (i != idx)
      {
        u=u->nextNode;
        i++;
      }
    }
    return u;
  } else
  {
    return NULL;
  }
}

void removeImpl(List* self, int idx)
{
  ListNode* remnode = self->get(self,idx);
  if (remnode->nextNode != NULL)
  {
    if (remnode->prevNode != NULL)
    {
      remnode->prevNode->nextNode = remnode->nextNode;
      remnode->nextNode->prevNode = remnode->prevNode;
      free(remnode);
    } else
    {
      remnode->nextNode->prevNode = NULL;
      free(remnode);
    }
  } else
  {
    if (remnode->prevNode == NULL)
    {
      free(remnode);
    } else
    {
      remnode->prevNode->nextNode = NULL;
      free(remnode);
    }
  }
  self->count--;
}

ListNode* popImpl(List* self, int idx)
{
  ListNode* remnode = self->get(self,idx);
  if (remnode->nextNode != NULL)
  {
    if (remnode->prevNode != NULL)
    {
      remnode->prevNode->nextNode = remnode->nextNode;
      remnode->nextNode->prevNode = remnode->prevNode;
    } else
    {
      remnode->nextNode->prevNode = NULL;
    }
  } else
  {
    if (remnode->prevNode != NULL)
    {
      remnode->prevNode->nextNode = NULL;
    }
  }
  self->count--;
  return remnode;
}

void appendImpl(List* self, ListNode* node)
{
    if (self->lastNode == NULL)
    {
      self->lastNode = node;
      self->firstNode = node;
      self->count = 1;
    } else
    {
      self->lastNode->nextNode = node;
      node->prevNode = self->lastNode;
      self->lastNode = node;
      self->count++;
    }
}

void insertImpl(List *self, ListNode* node, int idx)
{
  if (idx < self->count)
  {
    ListNode* insertBefore = self->get(self,idx);
    if (insertBefore->prevNode != NULL)
    {
      insertBefore->prevNode->nextNode = node;
      node->prevNode = insertBefore->prevNode;
    }
    insertBefore->prevNode = node;
    node->nextNode = insertBefore;
    if (idx==0)
    {
      self->firstNode = node;
    }
    self->count++;
  } else
  {
    self->append(self,node);
  }
}



List* NewList()
{
  List* list = (List*)malloc(sizeof(List));
  list->get = getImpl;
  list->remove = removeImpl;
  list->pop = popImpl;
  list->insert = insertImpl;
  list->append = appendImpl;
  list->count = 0;
  list->binarySearch = listBinarySearchImpl;
  list->firstNode = NULL;
  list->lastNode = NULL;
  return list;
}
