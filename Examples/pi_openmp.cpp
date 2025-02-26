// CPP Program to calculate PI using Leibniz equation
#include "omp.h"
#include <math.h>
#include <iomanip>
#include <iostream>

#define MAX 1000000000           // Number of iterations for leibniz calculation 
#define N_THREADS 8               // Maximum number of threads 
#define PAD 8

using namespace std; 
 
int main(){ 
    double sum[N_THREADS][PAD] = {0};
    double total_sum = 0.0;
    int batch_size = (int)(MAX/N_THREADS);

    #pragma omp parallel num_threads(N_THREADS)
    {
        int ID = omp_get_thread_num();
        int batch_start = batch_size * ID;
        int batch_end = batch_size * (ID + 1); 
        for(int i = batch_start; i < batch_end; ++i){
            sum[ID][0] += 4.0 / (2*i + 1);
            i++;
            sum[ID][0] -= 4.0 / (2*i + 1);
        } 
    }

    for(int i = 0; i < N_THREADS; ++i)
        total_sum += sum[i][0];

    cout << setprecision(10);
    cout << total_sum << endl;    

    return 0; 
} 