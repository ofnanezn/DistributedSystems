// CPP Program to find sum of array 
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <stdio.h> 

#define MAX 1000000			// size of array 
#define N_THREADS 2			// maximum number of threads 

float sum[N_THREADS] = { 0 }; 
int part = 0; 

void* leibniz_calculation(void* arg) { 
	// Each thread computes sum of 1/4th of array 
	int thread_part = part++; 

	for (int k = thread_part * (MAX / N_THREADS); k < (thread_part + 1) * (MAX / N_THREADS); k++) 
		sum[thread_part] += pow(-1, k) / (2*k + 1); 
} 

// Driver Code 
int main(){ 
	pthread_t threads[N_THREADS];
	clock_t t0 = clock(); 

	// Creating 4 threads 
	for (int i = 0; i < N_THREADS; i++) 
		pthread_create(&threads[i], NULL, leibniz_calculation, (void*)NULL); 

	// joining 4 threads i.e. waiting for all 4 threads to complete 
	for (int i = 0; i < N_THREADS; i++) 
		pthread_join(threads[i], NULL); 

	// adding sum of all 4 parts 
	float total_sum = 0; 
	for (int i = 0; i < N_THREADS; i++) 
		total_sum += sum[i]; 

	printf("Pi calculation = %f\n", 4 * total_sum); 
	printf("Time taken: %.2fs\n", (double)(clock() - t0)/CLOCKS_PER_SEC);

	return 0; 
} 

