// CPP Program to calculate PI using Leibniz equation
#include "omp.h"
#include <math.h>
#include <iomanip>
#include <iostream>

#define MAX 1000000000           // Number of iterations for leibniz calculation 
#define N_THREADS 2               // Maximum number of threads 

using namespace std; 
 
int main(){ 
    int *nums = new int[MAX];

    #pragma omp parallel for num_threads(N_THREADS)
        for(int i = 0; i < MAX; ++i){
            nums[i] = i;
        }

    #pragma omp parallel for num_threads(N_THREADS)
        for(int i = 0; i < MAX; ++i){
            nums[i] *= nums[i]; 
        } 

    for(int i = 0; i < 20; ++i)
        cout << nums[i] << endl;   

    delete []nums;

    return 0; 
} 