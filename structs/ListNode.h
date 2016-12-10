#include <stdlib.h>

typedef struct ListNode ListNode;

typedef void (* insertHereFunc)(ListNode* self, ListNode* n);

typedef struct {
  int ival;
  float fval;
  char* sval;
} dictKey;

struct ListNode
{
  void* value;
  dictKey key;
  ListNode* nextNode;
  ListNode* prevNode;
  insertHereFunc insertBefore;
  insertHereFunc insertAfter;
};

ListNode* NewListNode(void* value);
