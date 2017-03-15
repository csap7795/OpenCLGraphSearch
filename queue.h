#ifndef QUEUE_H_INCLUDED
#define QUEUE_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

struct node
{
    unsigned id;
    struct node* next;
};

typedef struct node node;

struct queue
{
    node* head;
    node* tail;
};

typedef struct queue queue;
void queue_add(queue* q,unsigned id);
unsigned queue_get(queue* q);
queue* init_queue ();
void free_queue(queue* q);
bool queue_is_empty(queue* q);

#endif // QUEUE_H_INCLUDED
