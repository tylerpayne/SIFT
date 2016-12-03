#include <stdlib.h>

typedef struct ListNode ListNode;

typedef void (* insertHereFunc)(ListNode* self, ListNode* n);

typedef struct {
  int ival;
  float fval;
  char* sval;
} keyUnion;

struct ListNode
{
  void* value;
  keyUnion key;
  ListNode* nextNode;
  ListNode* prevNode;
  insertHereFunc insertBefore;
  insertHereFunc insertAfter;
};

ListNode* NewListNode(void* value);
