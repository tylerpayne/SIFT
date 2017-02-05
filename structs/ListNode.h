#include <structs/Core.h>

#ifndef _LISTNODE_
#define _LISTNODE_

typedef struct ListNode ListNode;

typedef void (* insertHereFunc)(ListNode* self, ListNode* n);

struct ListNode
{
  void* value;
  Key key;
  ListNode* nextNode;
  ListNode* prevNode;
  insertHereFunc insertBefore;
  insertHereFunc insertAfter;
};

DLLEXPORT ListNode* NewListNode(void* value);
#endif
