#include <stdlib.h>
#include "ListNode.h"

typedef void (*appendFunc)(LinkedList*,ListNode*);
typedef void (*insertFunc)(LinkedList *, ListNode *, int);
typedef void (*getFunc)(LinkedList *, int);
typedef ListNode* (*popFunc)(LinkedList *, int);

typedef struct sLinkedList
{
  ListNode* firstNode;
  ListNode* lastNode;
  appendFunc append;
  insertFunc insert;
  getFunc remove;
  popFunc pop;
  getFunc get;
  int count;
} LinkedList;

LinkedList* NewLinkedList();
