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

ListNode* NewListNode(void* value);
