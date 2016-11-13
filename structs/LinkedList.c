#include "LinkedList.h"

ListNode* getImpl(LinkedList* self, int idx)
{
  if (idx < self->count)
  {
    if (idx <= self->count/2)
    {
      int i = 0;
      ListNode* u = self->firstNode;
      while (i != idx)
      {
        u=u->nextNode;
        i++
      }
      return u;
    } else
    {
      int i = self->count;
      ListNode* u = self->lastNode;
      while (i != idx)
      {
        u=u->prevNode;
        i--
      }
      return u;
    }
  } else
  {
      return NULL;
  }
}

void removeImpl(LinkedList* self, int idx)
{
  ListNode* remnode = self->get(idx);
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

ListNode* popImpl(LinkedList* self, int idx)
{
  ListNode* remnode = self->get(idx);
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
    if (remnode->prevNode == NULL)
    {
    } else
    {
      remnode->prevNode->nextNode = NULL;
    }
  }
  self->count--;
  return remnode;
}

void appendImpl(LinkedList* self, ListNode* node)
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
    }
}

void insertImpl(LinkedList *self, ListNode* node, int idx)
{
  if (idx < self->count)
  {
    ListNode *currentNode = getImpl(self,idx);
    currentNode->prevNode->nextNode = node;
    node->prevNode = currentNode->prevNode;
    currentNode->prevNode = node;
    node->nextNode = currentNode;
  }
}



LinkedList* NewLinkedList()
{
  LinkedList* list = (LinkedList*)malloc(sizeof(LinkedList));
  list->get = getImpl;
  list->remove = removeImpl;
  list->pop = popImpl;
  list->insert = insertImpl;
  list->append = appendImpl;
  return list;
}
