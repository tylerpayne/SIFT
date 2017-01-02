#include "List.c"

typedef struct Dictionary Dictionary;

typedef void (*addKeyValDictionaryFunc)(Dictionary*,char*,void*);
typedef void (*removeKeyDictionaryFunc)(Dictionary*, char*);
typedef void* (*getKeyDictionaryFunc)(Dictionary*, char*);
//char* array
typedef Array (*enumerateDictionaryKeysFunc)(Dictionary*);

struct Dictionary
{
  List* list;
  addKeyValKeypointFunc add;
  getKeyDictionaryFunc get;
  removeKeyDictionaryFunc remove;
  enumerateDictionaryKeysFunc enumerate;
};

Dictionary* NewDictionary();
