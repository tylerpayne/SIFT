#include "ListNode.h"

void insertBeforeImpl(ListNode* self, ListNode* n)
{
  if (self->prevNode == NULL)
  {
    self->prevNode = n;
    n->nextNode = self;
  } else
  {
    self->prevNode->nextNode = n;
    self->prevNode = n;
    n->nextNode = self;
  }
}

void insertAfterImpl(ListNode* self, ListNode* n)
{
  if (self->nextNode == NULL)
  {
    self->nextNode = n;
    n->prevNode = self;
  } else
  {
    self->nextNode->prevNode = n;
    self->nextNode = n;
    n->prevNode = self;
  }
}

ListNode* NewListNode(void* value)
{
  ListNode* node = (ListNode*)malloc(sizeof(ListNode));
  node->insertBefore = insertBeforeImpl;
  node->insertAfter = insertAfterImpl;
  node->value = value;
  node->nextNode = NULL;
  node->prevNode = NULL;
  node->key = NewIntKey(-1);
  return node;
}
