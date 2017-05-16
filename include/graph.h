#include <matrix.h>
#include <node.h>

#ifndef _GRAPH_H_
#define _GRAPH_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	Matrix *adjacency;
	Node **nodes;
} Graph;

#ifdef __cplusplus
}
#endif
#endif
