#include "queue.h"

void queue_add_beginning(queue* q, unsigned id)
{
    if(queue_is_empty(q))
        queue_add(q,id);

    node* n = (node*)malloc(sizeof(node));
    n->id = id;
    n->next = q->head;

    q->head = n;
}
void queue_add(queue* q,unsigned id)
{
    node* n = (node*)malloc(sizeof(node));
    n->id = id;
    n->next = NULL;

    if(q->head == NULL)
    {
        q->head = n;
    }
    if(q->tail == NULL)
    {
        q->tail = n;
    }
    else
    {
        node* current = q->tail;
        current->next = n;
        q->tail = n;
    }
}

unsigned queue_get(queue* q)
{
    if(q->head == NULL)
        return -1;

    if(q->head == q->head->next)
    {
        unsigned id =  q->head->id;
        free(q->head);
        q->head = NULL;
        q->tail = NULL;
        return id;
    }

    node* ret = q->head;
    unsigned id = ret->id;
    q->head = q->head->next;
    free(ret);
    return id;
}

queue* init_queue ()
{
    queue* q = (queue*) malloc(sizeof(queue));
    q->head = NULL;
    q->tail = NULL;

    return q;
}

void free_queue(queue* q)
{
    while(q->head != NULL)
    {
        queue_get(q);
    }
    free(q);
}

bool queue_is_empty(queue* q)
{
    return (q->head == NULL);
}
