#include <stdlib.h>

typedef void (* insertFunc)(ListNode* self, ListNode* n);

typedef struct sListNode
{
  void* value;
  void* key;
  ListNode* nextNode;
  ListNode* prevNode;
  insert insertBefore;
  insert insertAfter;
} ListNode;

ListNode* NewListNode(void* value);
ListNode* NewListNode(void* value, void* key);
