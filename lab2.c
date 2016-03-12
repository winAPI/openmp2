#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define DEBUG
// Компиляция: gcc lab2.c -o lab2 -fopenmp -std=c99

typedef struct tnode {
    int data;
    struct tnode* left;
    struct tnode* right;
} tnode;

void omp_info();
void quick_sort(int*, int, int);
void printf_array(int*, int);

void sum_tree_data(struct tnode*, int*);
tnode* create_node();
void free_tree(tnode*);

void test_tree()
{
    struct tnode* root = create_node();
    root->data = 20;
    root->left = create_node();
    root->left->data = 30;
    root->left->left = create_node();
    root->left->left->data = 10;
    root->left->right = create_node();
    root->left->right->data = 5;
    root->right = create_node();
    root->right->data = 40;
    
    
    int sum = 0;
    
    #pragma omp parallel shared(root, sum)
    #pragma omp single nowait 
    sum_tree_data(root, &sum);
    
    printf("sum = %d\n", sum);
    
    free_tree(root);
    free(root);
    
    /*     20
          /  \
         30  40
        /  \
       10   5
    */

    /*
    if(root->left)
        printf("root is NULL");
    printf("data=%d", root->data);
    free(root);
    */
}

void test_quick_sort()
{
    int n = 20;
    int* array = (int*)calloc(n, sizeof(int));
    
    time_t t;
    srand((unsigned) time(&t));
    for(int i = 0; i < n; ++i)
        array[i] = (rand() % 50);
    
    printf_array(array, n);
    
    int first = 0, last = n - 1;
    
    #pragma omp parallel shared(array)
    #pragma omp single nowait 
    quick_sort(array, first, last);
    
    printf_array(array, n);
    
    free(array);
}

int main(int argc, char** argv)
{
    test_tree();
    //test_quick_sort();
    
    return 0;
}

void quick_sort(int* array, int first, int last)
{
    int i = first, j = last; 
    
    int temp, x;
    
    x = array[first + (last - first) / 2]; // центральный элемент
    
    // процедура разделения
    do {
        while(array[i] < x) 
            i++;
        while(array[j] > x) 
            j--;

        if(i <= j) 
        {
            if(array[i] > array[j])
            {
                temp = array[i]; 
                array[i] = array[j]; 
                array[j] = temp;
            }
            
            i++; 
            j--;
        }
    } while(i <= j);

    #ifdef DEBUG
        #pragma omp critical (printf)
        printf("Thread # %d\n", omp_get_thread_num());
    #endif
    
    // рекурсивные вызовы
    #pragma omp task shared(array)
    if(i < last) quick_sort(array, i, last);
    #pragma omp task shared(array)
    if(first < j) quick_sort(array, first, j);
    #pragma omp taskwait
}

void printf_array(int* array, int length)
{
    for(int i = 0; i < length; ++i)
    {
        printf("%d, ", array[i]);
    }
    printf("\n\n");
}

void sum_tree_data(tnode* node, int* sum)
{
    if(node == NULL)
        return;
    
    #ifdef DEBUG
        #pragma omp critical (printf)
        printf("Thread # %d, elem=%d\n", omp_get_thread_num(), node->data);
    #endif
    
    #pragma omp critical (sum)
    *sum += node->data;
    
    #pragma omp task shared(node, sum)
    sum_tree_data(node->left, sum);
    #pragma omp task shared(node, sum)
    sum_tree_data(node->right, sum);
    #pragma omp taskwait
}

void free_tree(tnode* node)
{ 
    if(node == NULL)
        return;
    
    free_tree(node->left);
    free_tree(node->right);
    free(node->left);
    free(node->right);
    node->left = NULL;
    node->right = NULL;
}

tnode* create_node()
{
    return (tnode*)calloc(1, sizeof(tnode));
}

void omp_info()
{
    char* pEnv = getenv("OMP_NUM_THREADS");
    if (pEnv != NULL)
        printf("OMP_NUM_THREADS: %s\n", pEnv);
    pEnv = getenv("OMP_NUM_THREADS");
    if (pEnv != NULL)
        printf("Max threads in env: %s\n", pEnv);

    printf("Max threads = %d\n", omp_get_max_threads()); 
}