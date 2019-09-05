// CPP Program to calculate PI using Leibniz equation
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <chrono> 

#define MAX 1000000000			// Number of iterations for leibniz calculation 
#define N_THREADS 8				// Maximum number of threads 

using namespace std::chrono; 

double sum[N_THREADS] = { 0 }; 
int part = 0; 

void* leibniz_calculation(void* arg) { 
	// Each thread computes sum of 1/N_TREADS iterations
	int thread_part = part++; 

	for (int k = thread_part * (MAX / N_THREADS); k < (thread_part + 1) * (MAX / N_THREADS); k++) 
		sum[thread_part] += pow(-1, k) / (2*k + 1); 
} 

// Driver Code 
int main(){ 
	pthread_t threads[N_THREADS];
	auto start = high_resolution_clock::now();

	// Creating N_THREADS threads 
	for (int i = 0; i < N_THREADS; i++) 
		pthread_create(&threads[i], NULL, leibniz_calculation, (void*)NULL); 

	// joining N_THREADS threads i.e. waiting for all 4 threads to complete 
	for (int i = 0; i < N_THREADS; i++) 
		pthread_join(threads[i], NULL); 

	// adding sum of all N_THREADS parts 
	double total_sum = 0; 
	for (int i = 0; i < N_THREADS; i++) 
		total_sum += sum[i]; 

	printf("Pi calculation = %f\n", 4 * total_sum);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start); 

	std::cout << "Time taken: " << duration.count() * 1e-6 << " sec" << std::endl;

	return 0; 
} 

