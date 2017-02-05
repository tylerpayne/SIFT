#include <stdlib.h>
#include <stdio.h>
#include <structs/TreeNode.h>

#ifndef _HEAP_
#define _HEAP_

typedef struct Heap Heap;

typedef void (*addHeapFunc)(Heap*,TreeNode*);
typedef TreeNode* (*popHeapFunc)(Heap*);
typedef void (*heapifyFunc)(Heap*,Array);
typedef void (*reheapDownFunc)(Heap*);
typedef void (*reheapUpFunc)(Heap*);
typedef void (*swapHeapFunc)(Heap*,int,int);

struct Heap
{
  TreeNode* nodes;
  int size;
  int maxSize;
  addHeapFunc add;
  popHeapFunc pop;
  reheapDownFunc remove;
  heapifyFunc heapify;
  reheapUpFunc reheapUp;
  reheapDownFunc reheapDown;
  swapHeapFunc swap;
};

DLLEXPORT Heap* NewHeap(int maxSize);

#endif
