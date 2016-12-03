#include <stdlib.h>
#include "ListNode.c"

typedef struct List List;

typedef void (*vlllnFunc)(List*,ListNode*);
typedef void (*vlllniFunc)(List*, ListNode*, int);
typedef void (*vlliFunc)(List*,int);
typedef ListNode* (*lllluFunc)(List*, keyUnion);
typedef ListNode* (*lllliFunc)(List*, int);
typedef ListNode* (*llllsFunc)(List*, char*);

struct List
{
  ListNode* firstNode;
  ListNode* lastNode;
  vlllnFunc append;
  vlllniFunc insert;
  vlliFunc remove;
  lllliFunc pop;
  lllluFunc get;
  lllliFunc getIndex;
  llllsFunc getKey;
  int count;
};

List* NewList();
