typedef struct Heap Heap;

typedef void (*enQHeapFunc)(Heap*,ListNode*);
typedef ListNode* (*deQHeapFunc)(Heap*);
typedef void (*heapifyFunc)(Heap*,ListNode*);
typedef void (*reheapDownFunc)(Heap*);
typedef void (*reheapUpFunc)(Heap*);

struct Heap
{
  int size;
  ListNode* nodes;
  enQHeapFunc enQ;
  deQHeapFunc deQ;
  heapifyFunc heapify;
  reheapUpFunc reheapUp;
  reheapDownFunc reheapDown;
};

Heap* NewHeap();
