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

Heap* NewHeap(int maxSize);
