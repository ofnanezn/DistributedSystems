// CPP Program to calculate PI using Leibniz equation
#include "omp.h"
#include <math.h>
#include <iomanip>
#include <iostream>

#define MAX 1000000000           // Number of iterations for leibniz calculation 
#define N_THREADS 16               // Maximum number of threads 

using namespace std; 
 
int main(){ 
    double total_sum = 0.0;

    #pragma omp parallel for num_threads(N_THREADS) reduction(+:total_sum)
    for(int i = 0; i < MAX; ++i)
        total_sum += 4.0 * (i % 2 == 0 ? 1: -1) / (2.0 * i + 1); 

    cout << setprecision(10);
    cout << total_sum << endl;    

    return 0; 
} 