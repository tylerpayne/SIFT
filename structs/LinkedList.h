#include <stdlib.h>
#include "ListNode.c"

typedef struct List List;

typedef void (*vlllnFunc)(List*,ListNode*);
typedef void (*vlllniFunc)(List*, ListNode*, int);
typedef void (*vlliFunc)(List*,int);
typedef ListNode* (*lllliFunc)(List*, int);
typedef ListNode* (*llllfFunc)(List*, float);

struct List
{
  ListNode* firstNode;
  ListNode* lastNode;
  vlllnFunc append;
  vlllniFunc insert;
  vlliFunc remove;
  lllliFunc pop;
  lllliFunc get;
  llllfFunc binarySearch;
  int count;
  float* keyArray;
  ListNode* nodePtrArray;
};

List* NewList();
