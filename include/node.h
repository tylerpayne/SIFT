#include <core.h>

#ifndef _NODE_H_
#define _NODE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Node Node;

struct Node
{
	int adj_len;
	void *data;
	Key key;
	Node **adjacent;
};

#include <graph_funcs.h>

#ifdef __cplusplus
}
#endif

#endif
