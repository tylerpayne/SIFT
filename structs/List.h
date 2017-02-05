#include <structs/ListNode.h>
#include <string.h>

#ifndef _LIST_
#define _LIST_

typedef struct List List;

typedef void (*vlllnFunc)(List*,ListNode*);
typedef void (*vlllniFunc)(List*, ListNode*, int);
typedef void (*vlliFunc)(List*,int);
typedef ListNode* (*lllluFunc)(List*, Key);
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

DLLEXPORT List* NewList();
#endif
