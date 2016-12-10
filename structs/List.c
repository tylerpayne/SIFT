#include "List.h"
#include <string.h>

ListNode* getSImpl(List* self, char* key)
{
  if (self->count > 0)
  {
    ListNode* n = self->firstNode;
    while (strcmp(n->key.sval,key) == 1)
    {
      if (n->nextNode == NULL)
      {
        n = NULL;
        break;
      }
      n = n->nextNode;
    }
    return n;
  } else
  {
    return NULL;
  }
}

ListNode* getIImpl(List* self, int idx)
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

ListNode* getFImpl(List* self, float key)
{
  if (self->count > 0)
  {
    //TO DO
    return NULL;
  } else
  {
    return NULL;
  }
}

ListNode* getImpl(List* self, dictKey idx)
{
  if (idx.sval != NULL)
  {
    return getSImpl(self,idx.sval);
  }
  return getIImpl(self,idx.ival);
}


void removeImpl(List* self, int idx)
{
  ListNode* remnode = self->getIndex(self,idx);
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
  ListNode* remnode = self->getIndex(self,idx);
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
      node->key.ival = 0;
      self->lastNode = node;
      self->firstNode = node;
      self->count = 1;
    } else
    {
      self->lastNode->nextNode = node;
      node->prevNode = self->lastNode;
      self->lastNode = node;
      node->key.ival = self->count;
      self->count++;
    }
}

void insertImpl(List *self, ListNode* node, int idx)
{
  if (idx < self->count)
  {
    ListNode* insertBefore = self->getIndex(self,idx);
    if (insertBefore->prevNode != NULL)
    {
      insertBefore->prevNode->nextNode = node;
      node->prevNode = insertBefore->prevNode;
    }
    insertBefore->prevNode = node;
    node->nextNode = insertBefore;
    node->key.ival = idx;
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
  list->getIndex = getIImpl;
  list->getKey = getSImpl;
  list->remove = removeImpl;
  list->pop = popImpl;
  list->insert = insertImpl;
  list->append = appendImpl;
  list->count = 0;
  //list->binarySearch = listBinarySearchImpl;
  list->firstNode = NULL;
  list->lastNode = NULL;
  return list;
}
