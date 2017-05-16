#include <graph.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <queue.h>
#include <alloca.h>

void addEdges(Graph* graph, cl_uint* src, cl_uint* dest, unsigned length, cl_float* weight)
{
    //Calculate new indices for vertices
    int j = 0;
    int i = 0;
    while(j<length)
    {
        while(i<=src[j])
        {
            graph->vertices[i] += j;
            i++;
        }
        j++;
    }
    while(i<=graph->V)
    {
        graph->vertices[i] += j;
        i++;
    }

    // calculate number of edges and allocate new buffers for edges and weights
    graph->E += length;
    cl_uint* new_edges = (cl_uint*)malloc(graph->E*sizeof(cl_uint));
    cl_float* new_weights = (cl_float*)malloc(graph->E*sizeof(cl_float));

    // copy old data to new and add new edges
    i = 0;
    j = 0;
    while(j<length)
    {
        // Copy the old data
        while(i<graph->vertices[src[j]+1]-1)
        {
            new_edges[i] = graph->edges[i-j];
            new_weights[i] = graph->weight[i-j];
            i++;
        }
        // add the new data
        new_edges[i] = dest[j];
        new_weights[i] = weight[j];
        i++;
        //copy the next old datas
        unsigned end;
        if(j<length-1)
            end = graph->vertices[src[j+1]];
        else
            end = graph->E;

        while(i<end)
        {
            new_edges[i] = graph->edges[i-j-1];
            new_weights[i] = graph->weight[i-j-1];
            i++;
        }
        j++;

    }

    // free old data
    free(graph->edges);
    free(graph->weight);

    // set pointers to new edge & weight array
    graph->edges = new_edges;
    graph->weight = new_weights;
}
void addEdge(Graph* graph, cl_uint src, cl_uint dest, cl_float weight)
{
    for(int i = src; i<graph->V;i++)
    {
        graph->vertices[i+1]++;
    }

    graph->E++;
    cl_uint* new_edges = (cl_uint*)malloc(graph->E*sizeof(cl_uint));
    cl_float* new_weights = (cl_float*)malloc(graph->E*sizeof(cl_float));

    int i;
    for(i = 0; i<graph->vertices[src+1]-1;i++)
    {
        new_edges[i] = graph->edges[i];
        new_weights[i] = graph->weight[i];
    }
    new_edges[i] = dest;
    new_weights[i] = weight;
    for(i = graph->vertices[src+1];i<graph->E;i++)
    {
        new_edges[i] = graph->edges[i-1];
        new_weights[i] = graph->weight[i-1];
    }
    free(graph->edges);
    free(graph->weight);

    graph->edges = new_edges;
    graph->weight = new_weights;
}
void connectGraphbfs(Graph* graph)
{
    srand(time(NULL));
    unsigned *components = (unsigned*)calloc(graph->V,sizeof(unsigned));
    cl_uint *src = (cl_uint*)calloc(graph->V,sizeof(cl_uint));
    cl_uint *dest = (cl_uint*)calloc(graph->V,sizeof(cl_uint));
    cl_float *weight = (cl_float*)calloc(graph->V,sizeof(cl_float));
    unsigned count = 0;

    for(int j = 0; j<graph->V;j++)
    {
        if(components[j] == 1)
            continue;

        if(j != 0)
        {
           //addEdge(graph,j-1,j,(cl_float)((double)rand()/(double)RAND_MAX));
            src[count] = j-1;
            dest[count] = j;
            weight[count] = (cl_float)((double)rand()/(double)RAND_MAX);
            count++;
        }

        queue* to_watch = init_queue();
        queue_add(to_watch,j);
        while(!queue_is_empty(to_watch)){
            unsigned node = queue_get(to_watch);
            components[node] = 1;
            for(int i = graph->vertices[node]; i <graph->vertices[node+1];i++)
            {
                unsigned neighbor = graph->edges[i];
                if(components[neighbor] == 0)
                    queue_add(to_watch,neighbor);
            }
        }
        free_queue(to_watch);
    }
    if(count != 0)
        addEdges(graph,src,dest,count,weight);

    free(src);
    free(dest);
    free(weight);

    /*for(int i = 0; i<graph->V-1;i++)
    {
        if(components[i+1] != 1)
        {
            return;
        }
    }*/
    free(components);
    if(count != 0)
        connectGraphbfs(graph);
}

void connectGraph(Graph* graph)
{
    srand(time(NULL));
    unsigned *components = (unsigned*)calloc(graph->V,sizeof(unsigned));
    dfs(graph,components,0,1);
    for(int i = 1; i<graph->V;i++)
    {
        if(components[i] == 0)
        {
            addEdge(graph,i-1,i,(cl_float)((double)rand()/(double)RAND_MAX));
            dfs(graph,components,i,1);
        }
    }
    for(int i = 0; i<graph->V-1;i++)
    {
        if(components[i+1] != components[i])
        {
            return;
        }
    }
    free(components);
}

void dfs(Graph* graph, unsigned* components, cl_uint v, unsigned c)
{
    if(components[v] != 0)
        return;

    components[v] = c;
    for(int i = graph->vertices[v];i<graph->vertices[v+1];i++)
    {
        dfs(graph,components,graph->edges[i],c);
    }
}
Graph* getSemaphoreGraph(int verticeCount)
{

    srand(time(NULL));
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->vertices = (cl_uint*) malloc(sizeof(cl_uint)*(verticeCount+1));

    graph->V = verticeCount;
    graph->E = 2*(verticeCount-2);
    graph->edges = (cl_uint*)malloc(sizeof(cl_uint)*graph->E);
    graph->weight = (float*)malloc(sizeof(float)*graph->E);

    // FILL VERTICES
    graph->vertices[0] = 0;
    graph->vertices[1] = verticeCount-2;
    for(int i = 2; i < verticeCount; i++ )
    {
        graph->vertices[i] = graph->vertices[i-1]+1;
    }

    // FILL EDGES and WEIGHTS ( NOTE THE AMOUNT OF EDGES WILL BE A MULTIPLE OF 2 AS THE GRAPH IS SYMETRIC
    int i;
    for(i = 0; i<graph->E/2;i++)
    {
        graph->edges[i] = i+1;
        graph->weight[i] = (float)((double)rand()/(double)RAND_MAX);;
    }
    for(int j = graph->E/2;j<graph->E;j++)
    {
        graph->edges[j] = i+1;
        graph->weight[j] = (float)((double)rand()/(double)RAND_MAX);;
    }
    //Make sure there exists a shortes path , in this case : in the middle
    graph->weight[graph->E/4] = 0.0f;
    graph->weight[graph->E/2+graph->E/4] = 0.0f;
        graph->vertices[verticeCount]=graph->E;
    return graph;

}

Graph* createUnconnectedGraph()
{
    Graph* graph = getEmptyGraph(11,10);
    for(int i = 0; i<graph->V-1;i++)
    {
        graph->vertices[i] = i;
    }
    graph->vertices[graph->V-1] = graph->E;
    graph->vertices[graph->V] = graph->E;
    for(int i = 0; i<graph->E;i++)
    {
        if(i<graph->E/2)
        {
            graph->edges[i] = (i+1)%(graph->E/2);
            graph->weight[i] = 0;
        }
        else if(i != graph->E-1)
        {
            graph->edges[i] = i+1;
            graph->weight[i] = 0;
        }
        else
        {
            graph->edges[i] = graph->E/2;
            graph->weight[i] = 0;
        }
    }
    return graph;
}

Graph* matrixToGraph(cl_float** matrix, int length)
{
    //Count Edges
    unsigned edgecount = 0;
    for(int i = 0; i< length;i++)
        for(int j = 0; j<length;j++)
            if(matrix[i][j] != CL_FLT_MAX && matrix[i][j] != 0)
                edgecount++;

    Graph* graph = getEmptyGraph(length,edgecount);

    //fill the graph
    edgecount = 0;
    graph->vertices[0] = 0;
    for(int i = 0; i<length-1;i++)
    {
        for(int j = 0; j<length;j++)
            if(matrix[i][j] != CL_FLT_MAX && matrix[i][j] != 0)
            {
                graph->edges[edgecount] = j;
                graph->weight[edgecount] = matrix[i][j];
                graph->vertices[i+1] = ++edgecount;

            }
    }
    // for the last row, only calculate the edges
    for(int j = 0; j<length;j++)
    {
        if(matrix[length-1][j] != CL_FLT_MAX)
        {
            graph->edges[edgecount] = j;
            graph->weight[edgecount] = matrix[length-1][j];
            edgecount++;
        }
    }

    return graph;

}

Graph* createNegativeCycleGraph(unsigned vertices)
{
    Graph* graph = getEmptyGraph(vertices,vertices);
    for(int i = 0; i<graph->V;i++)
    {
        graph->vertices[i] = i;
        graph->edges[i] = (i+1) % graph->V;
        graph->weight[i] = -1.0f;
    }
    return graph;
}

Graph* getEmptyGraph(unsigned vertices, unsigned edges)
{
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->E = edges;
    graph->V = vertices;
    graph->edges = (cl_uint*)calloc(graph->E,sizeof(cl_uint));
    graph->vertices = (cl_uint*)calloc((graph->V+1),sizeof(cl_uint));
    graph->weight = (cl_float*) calloc(graph->E , sizeof(cl_float));
    return graph;
}

Graph* getTreeGraphWeight(int level, int edges)
{
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = 0;

    for(int i = 0 ; i<level;i++)
    {
        graph->V += (unsigned)pow((float)edges,i);
    }

    graph->vertices = (unsigned*) malloc(sizeof(unsigned)*(graph->V+1));
    graph->E = graph->V-1;
    graph->edges = (unsigned*)malloc(sizeof(unsigned)*graph->E);
    graph->weight = (float*)malloc(sizeof(float)*graph->E);


    for(int i = 0; i<(graph->V)/edges;i++)
    {
       graph->vertices[i] = edges*i;
       graph->edges[i] = i+1;
       graph->weight[i] = 1.0f;
    }

    for(int i = (graph->V)/edges;i<graph->V;i++)
    {
        graph->vertices[i] = graph->E;
        if(i<graph->E){
            graph->edges[i] = i+1;
            graph->weight[i] = 1.0f;
        }
    }
    // Last Entry is number of edges
    graph->vertices[graph->V] = graph->E;

    return graph;
}

Graph* getRandomTreeGraph(int level, int edges, unsigned edgeperVertex)
{
    //initialize random generator
    srand(time(NULL));

    //allocate Graph Data
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = 0;

    // Calculate number of vertices
    for(int i = 0 ; i<level;i++)
    {
        graph->V += (unsigned)pow((float)edges,i);
    }

    graph->vertices = (cl_uint*) malloc(sizeof(cl_uint)*(graph->V+1));

    //set first two values for start node
    graph->vertices[0] = 0;
    graph->vertices[1] = edges;

    graph->E = edges;

    cl_uint start_node = 0;
    for(int i = 1; i<=level-1;i++)
    {
        start_node++;
        unsigned level_nodes = (unsigned)pow((float)edges,i);
        unsigned end_node = level_nodes + start_node -1;

        for(int j = start_node;j<=end_node;j++)
        {
            unsigned num_edges;
            if(i == level-1)
                num_edges = 0;
            else
            {
                if(pow((float)edges,i+1)<edgeperVertex)
                    num_edges = rand()%(unsigned)pow((float)edges,i+1);
                else
                    num_edges = rand()%edgeperVertex;
            }

            graph->vertices[j+1] = graph->vertices[j]+num_edges;
            graph->E += num_edges;
        }
        start_node = end_node;
    }

    //allocate data vor edges
    graph->edges = (cl_uint*) malloc(sizeof(cl_uint)*(graph->E));
    // set first "edges" edges
    for(int i = 0; i<edges;i++)
        graph->edges[i] = i+1;

    start_node = 0;
    for(int i = 1; i<level-1;i++)
    {
        start_node++;
        unsigned level_nodes = (unsigned)pow((float)edges,i);
        unsigned end_node = level_nodes + start_node -1;

        for(int j = start_node;j<=end_node;j++)
        {
            cl_uint length = graph->vertices[j+1] - graph->vertices[j];
            assignRandomNumbersNotTheSame(&graph->edges[graph->vertices[j]],length,end_node+1,(unsigned)pow((float)edges,i+1));
        }
        start_node = end_node;
    }

    graph->weight = NULL;
    return graph;

}

void assignRandomNumbersNotTheSame(cl_uint *edges, cl_uint length, unsigned start_node, unsigned num_edges)
{
    unsigned* neighbors = (unsigned*)alloca(sizeof(unsigned)*length);
    for(int i = 0; i<length;i++)
    {
        unsigned next_num = (rand()%num_edges) + start_node;
        int j = 0;
        while(j<i)
        {
            if(next_num == neighbors[j])
                next_num + 1 < (start_node+num_edges) ? (next_num++) : (next_num = start_node);

            j++;
        }
        neighbors[i] = next_num;
        edges[i] = next_num;
    }
}

Graph* getTreeGraph(int level, int edges)
{
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = 0;

    for(int i = 0 ; i<level;i++)
    {
        graph->V += (unsigned)pow((float)edges,i);
    }

    graph->vertices = (unsigned*) malloc(sizeof(unsigned)*(graph->V+1));
    graph->E = graph->V-1;
    graph->edges = (unsigned*)malloc(sizeof(unsigned)*graph->E);
    graph->weight = NULL;


    for(int i = 0; i<(graph->V)/edges;i++)
    {
       graph->vertices[i] = edges*i;
       graph->edges[i] = i+1;
    }

    for(int i = (graph->V)/edges;i<graph->V;i++)
    {
        graph->vertices[i] = graph->E;
        if(i<graph->E)
            graph->edges[i] = i+1;
    }
    // Last Entry is number of edges
    graph->vertices[graph->V] = graph->E;

    return graph;
}

bool graph_equal(Graph* g1, Graph* g2)
{
    if(g1->V != g2->V || g1->E != g2->E)
        return false;

    for(int i = 0; i<g1->V;i++)
        if(g1->vertices[i] != g2->vertices[i])
            return false;

    // check sums
    for(int i = 0; i<g1->E;i++)
        if(g1->edges[i] != g2->edges[i])
            return false;
    // check sums
    for(int i = 0; i<g1->E;i++)
        if(g1->weight[i] != g2->weight[i])
            return false;

    return true;

}
unsigned getNormalDistributedValues(unsigned range)
{
    unsigned repeats = 16;
    unsigned ret_val = 0;
    for(int i = 0; i<repeats;i++)
    {
        ret_val += rand()%range;
    }
    ret_val = (unsigned)(ret_val / (float)repeats + 0.5);
    return ret_val;
}

Graph* getRandomGraph (unsigned verticeCount, unsigned edges_per_vertex){

    srand(time(NULL));

    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->vertices = (cl_uint*) malloc(sizeof(cl_uint)*(verticeCount+1));
    graph->V = verticeCount;

    unsigned edgeCount = 0;

    // generate random graph : 1.Step : Each Vertice gets at maximum edges_per_vertex -1 edges
    for(int i = 0;i<verticeCount;i++)
    {
        graph->vertices[i] = edgeCount;
        edgeCount += getNormalDistributedValues(edges_per_vertex);
    }

    // Last Entry is number of edges
    graph->E = edgeCount;
    graph->vertices[graph->V] = graph->E;


    // create data for Edges and Weight
    graph->edges = (cl_uint*) malloc(sizeof(cl_uint)*graph->E);
    graph->weight = (cl_float*) malloc(sizeof(cl_float)*graph->E);

    // 2.Step : Each Edge gets a Weight-Value between 0 and 1
    unsigned i = 0;
    for(i = 0; i<graph->V;i++){
        unsigned *dest = (unsigned*)malloc(sizeof(unsigned)*(graph->vertices[i+1]-graph->vertices[i]));
        bool flag;
        for(int j = graph->vertices[i]; j<graph->vertices[i+1];j++)
        {
            flag = true;
            unsigned node = (unsigned )rand()%graph->V;
            // Make sure two nodes are only connected by one edge
            while(flag)
            {
                flag = false;
                for(int k = 0; k<j-graph->vertices[i];k++)
                {
                    if(dest[k] == node)
                    {
                        flag = true;
                        node = (node+1)%graph->V;
                    }
                }

            }
            if(node != i){
                graph->edges[j] = node;
                graph->weight[j] = (cl_float)((double)rand()/(double)RAND_MAX);
                dest[j-graph->vertices[i]] = node;
            }
            else
                j--;
        }
    }



    return graph;

}

//Frees allocated Space for Graph Data
void freeGraph(Graph* graph)
{
    free(graph->vertices);
    free(graph->edges);
    free(graph->weight);
    free (graph);
}

//prints Graph to Console
void printGraph(Graph* graph)
{
    printf("VERTICES[%u]:\t",graph->V);

    for(int i = 0; i<graph->V; i++)
        printf("%u\t",graph->vertices[i]);

    printf("\n");

    printf("EDGES[%u]:\t",graph->E);

    for(int i = 0; i<graph->E; i++)
        printf("%u\t",graph->edges[i]);

    printf("\n");

    if(graph->weight != NULL)
    {
        printf("WEIGHT[%u]:\t",graph->E);

        for(int i = 0; i<graph->E; i++)
            printf("%.1f\t",graph->weight[i]);

        printf("\n");
    }

}



Graph* readGraphFromFile(const char* filename)
{
    FILE *fp = fopen(filename,"r");
    unsigned vertice_count;
    unsigned edge_count;

    // reads first line of input data, 2 values, one is amount of vertices, one amount of edges
    fscanf(fp, "%u", &vertice_count);
    fscanf(fp, "%u", &edge_count);
    fscanf(fp, "\n");

    Graph* graph = getEmptyGraph(vertice_count,edge_count);

    //read vertices
    fread(graph->vertices,sizeof(graph->vertices[0]),graph->V, fp) ;
    fscanf(fp, "\n");

    //read edges
    fread(graph->edges,sizeof(graph->edges[0]),graph->E, fp) ;
    fscanf(fp, "\n");

    //read weight
    fread(graph->weight,sizeof(graph->weight[0]),graph->E, fp) ;
    fclose(fp);

    // Last Entry is number of edges
    graph->vertices[graph->V] = graph->E;

    return graph;
}


void writeGraphToFile(const char *filename, Graph* graph)
{
    FILE *fp = fopen(filename,"w");

    fprintf(fp,"%u\t%u\n",graph->V,graph->E);

    //write vertices
    fwrite(graph->vertices,sizeof(graph->vertices[0]),graph->V, fp) ;
    fprintf(fp,"\n");

    //write edges
    fwrite(graph->edges,sizeof(graph->edges[0]),graph->E, fp) ;
    fprintf(fp,"\n");

    //write weight
    fwrite(graph->weight,sizeof(graph->weight[0]),graph->E, fp) ;
    fclose(fp);

}

bool checkacyclic(Graph* graph)
{
    queue* q = init_queue();
    bool* visited = (bool*)calloc(graph->V,sizeof(bool));
    unsigned* count_neighbor = (unsigned*)calloc(graph->V,sizeof(unsigned));

    int source = 0;
    do
    {

        visited[source] = true;
        unsigned num_neighbors = graph->vertices[source+1]-graph->vertices[source];
        if(count_neighbor[source] < num_neighbors)
        {
            queue_add_beginning(q,source);
            unsigned neighbor = graph->edges[graph->vertices[source]+ count_neighbor[source]];
            if(!visited[neighbor]){
                count_neighbor[source]++;
                source = neighbor;
            }
            else
            {
                free(visited);
                free(count_neighbor);
                free_queue(q);
                return false;
            }
        }

        else
        {
        visited[source] = false;
        source = queue_get(q);
        }

    }while(!queue_is_empty(q));



    free(visited);
    free(count_neighbor);
    free_queue(q);
    return true;
}

Graph* removeCycles(Graph* graph)
{
    queue* q = init_queue();
    bool* visited = (bool*)calloc(graph->V,sizeof(bool));
    unsigned* count_neighbor = (unsigned*)calloc(graph->V,sizeof(unsigned));

    int source = 0;
    int max = 0;
    int maximum =0;
    do
    {

        visited[source] = true;
        unsigned num_neighbors = graph->vertices[source+1]-graph->vertices[source];
        if(count_neighbor[source] < num_neighbors)
        {
            queue_add_beginning(q,source);
            unsigned neighbor = graph->edges[graph->vertices[source]+ count_neighbor[source]];
            if(!visited[neighbor]){
                count_neighbor[source]++;
                max++;
                source = neighbor;
            }
            else
            {
                graph->edges[graph->vertices[source]+ count_neighbor[source]] = CL_UINT_MAX;
                count_neighbor[source]++;
            }
        }

        else
        {
        visited[source] = false;
        if(max>maximum)
            maximum = max;
        max--;
        source = queue_get(q);
        }

    }while(!queue_is_empty(q));


    printf("Max: %d\n",maximum);
    free(visited);
    free(count_neighbor);
    free_queue(q);
    return createAcyclicFromGraph(graph);


}

void removeCycleSource(Graph* graph, unsigned source)
{
    queue* q = init_queue();
    bool* visited = (bool*)calloc(graph->V,sizeof(bool));

    for(int i = graph->vertices[source];i<graph->vertices[source+1];i++)
        if(graph->edges[i] != CL_UINT_MAX)
            queue_add(q,graph->edges[i]);


    while(!queue_is_empty(q))
    {
        unsigned node = queue_get(q);
        for(int i = graph->vertices[node];i<graph->vertices[node+1];i++)
        {
            unsigned e = graph->edges[i];
            if(e == CL_UINT_MAX)
                continue;

            if(e == source)
                graph->edges[i] = CL_UINT_MAX;

            else if (visited[e] == false)   {
                queue_add(q,e);
                visited[e] = true;
            }

        }
    }
    free(visited);
    free_queue(q);
}


Graph* createAcyclicFromGraph(Graph* graph)
{
    unsigned nRemoved = 0;
    for(int i = 0; i<graph->E;i++)
    {
        if(graph->edges[i] == CL_UINT_MAX)
            nRemoved++;
    }
    Graph* acyclic = getEmptyGraph(graph->V,graph->E-nRemoved);

    if(graph->weight == NULL)
    {
        free(acyclic->weight);
        acyclic->weight = NULL;
    }

    unsigned edge_count = 0;
    for(int i = 0; i<graph->V;i++)
    {
        acyclic->vertices[i] = edge_count;
        for(cl_uint e = graph->vertices[i]; e < graph->vertices[i+1]; e++)
        {
            if(graph->edges[e] != CL_UINT_MAX)
            {
                acyclic->edges[edge_count] = graph->edges[e];

                if(graph->weight != NULL)
                    acyclic->weight[edge_count] = graph->weight[e];

                edge_count++;
            }
        }
    }
    acyclic->vertices[acyclic->V] = acyclic->E;
    freeGraph(graph);
    return acyclic;
}

Graph* readInTextData(const char* filename)
{
    FILE *fp = fopen(filename,"r");
    Graph* graph = getEmptyGraph(1632803,30622564);
    free(graph->weight);
    unsigned old = 0;
    unsigned rec = 0;
    unsigned neighbor;
    unsigned edgecount = 0;
    graph->vertices[0] = 0;
    while(fscanf(fp, "%u%u",&rec,&neighbor) == 2)
    {
        if(rec-1 != old)
        {
            old = rec-1;
            graph->vertices[old] = edgecount;
            graph->edges[edgecount] = neighbor-1;
        }
        else
        {
            graph->vertices[old] = edgecount;
            graph->edges[edgecount] = neighbor-1;
        }
        edgecount++;
    }
    free(fp);
    return graph;
}

void parseFile(const char* filename, int** vertices, int** edges, int** weight,  int* vertice_count,  int* edge_count, int* weight_count)
{
    /*FILE *fp = fopen(filename,"r");
    int amount;

    // reads input data from file for vertices and stores them in vertices
    fscanf(fp, "%d", vertice_count);
    int* vertice_buffer = (int*) realloc(*vertices,*vertice_count * sizeof(int));

    for (int i = 0; i < *vertice_count; i++)
    {
        amount = fscanf(fp, "%d", &vertice_buffer[i]);
        //printf("vertice_buffer[%d] = %d\n", i , vertices[i]);
    }

    *vertices = vertice_buffer;

    /* reads input data from file for edges and stores them in edges
    fscanf(fp, "%d", edge_count);
    int* edge_buffer = (int*) realloc(*edges,*edge_count * sizeof(int));

    for (int i = 0; i < *edge_count; i++)
    {
        amount = fscanf(fp, "%d", &edge_buffer[i]);
    }

    *edges = edge_buffer;

    /* reads input data from file for weight and stores them in weight
    fscanf(fp, "%d",weight_count);
    int* weight_buffer = (int*) realloc(*weight,*weight_count * sizeof(int));

    for (int i = 0; i < *weight_count; i++)
    {
        amount = fscanf(fp, "%d", &weight_buffer[i]);
    }
    *weight = weight_buffer;


    fclose(fp);*/
}




