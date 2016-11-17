#include <stdlib.h>
#include "ListNode.c"

typedef struct LinkedList LinkedList;

typedef void (*appendFunc)(LinkedList*,ListNode*);
typedef void (*insertFunc)(LinkedList*, ListNode*, int);
typedef void (*removeFunc)(LinkedList*,int);
typedef ListNode* (*getFunc)(LinkedList*, int);
typedef ListNode* (*popFunc)(LinkedList*, int);

struct LinkedList
{
  ListNode* firstNode;
  ListNode* lastNode;
  appendFunc append;
  insertFunc insert;
  removeFunc remove;
  popFunc pop;
  getFunc get;
  int count;
};

LinkedList* NewLinkedList();
