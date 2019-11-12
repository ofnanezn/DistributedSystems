/************* add vector ******************************************************/
#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define PI 3.14159265359

  
/*******************************************************************************/
__global__ void initialize_pop(float *total_pop, int num_elements){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < num_elements){
		curandState state;
        curand_init(clock64(), i, 0, &state);
        total_pop[i] = curand_uniform(&state);
	}
}

__global__ void calculate_fitness(const float *total_pop, float *total_fitness, int dims, int num_elements){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements){
    	total_fitness[i] = 10.0 * dims;

		for(int j = i * dims; j < i * (dims + 1); ++j)
			total_fitness[i] += (total_pop[j]*total_pop[j] - 10 * cos(2.0 * PI * total_pop[j]));
    }	
}

__global__ void individual_evolution(float *total_pop, float *total_fitness, int dims, int pop_size, int n_islands, int num_elements){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements){
    	int num_individuals = pop_size * n_islands;

      curandState state;
      curand_init(clock64(), i, 0, &state);
          
      int parent2_pos = (int)(curand_uniform(&state) * num_individuals);
      
      // Xover
      int crosspoint = (int)(curand_uniform(&state)*(dims - 1) + 1);
    }	
}


/*******************************************************************************/
int main(void)
{

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    int dims = 2, pop_size = 100, n_islands = 10;

	/******************************** 
	Population initialization
	*********************************/
    // Print the vector length to be used, and compute its size
    int total_pop_size = dims * pop_size * n_islands;
    size_t total_size = total_pop_size * sizeof(float);
    printf("[Island Genetic Algorithm with %d elements]\n", total_pop_size);

    float *h_total_pop = (float *)malloc(total_size);
    
    if(h_total_pop == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    float *d_total_pop = NULL; 
    err = cudaMalloc((void **) &d_total_pop, total_size);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector for total population (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_total_pop, h_total_pop, total_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid =(total_pop_size + threadsPerBlock - 1) / threadsPerBlock;

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    initialize_pop<<<blocksPerGrid, threadsPerBlock>>>(d_total_pop, total_pop_size);

    err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_total_pop, d_total_pop, total_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*for(int i = 0; i < 10; ++i){
    	printf("%d\n", h_total_pop[i]);
    }*/


    /******************************** 
	Fitness evaluation
	*********************************/
    int fitness_elements = pop_size * n_islands;
    size_t total_fitness_size = fitness_elements * sizeof(float);

    float *h_total_fitness = (float *)malloc(total_fitness_size);

    if (h_total_pop == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    float *d_total_fitness = NULL; 
    err = cudaMalloc((void **) &d_total_fitness, total_fitness_size);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector for total population (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_total_fitness, h_total_fitness, total_fitness_size, cudaMemcpyHostToDevice);

    threadsPerBlock = 128;
    blocksPerGrid = (fitness_elements + threadsPerBlock - 1) / threadsPerBlock;

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    calculate_fitness<<<blocksPerGrid, threadsPerBlock>>>(d_total_pop, d_total_fitness, dims, fitness_elements);

    err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_total_fitness, d_total_fitness, total_fitness_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    //float *child1, *child2;
    //float 

    for(int i = 0; i < 10; ++i)
    	printf("%f\n", h_total_fitness[i]);


    /******************************** 
	Free memory
	*********************************/
    // Free device global memory
    err = cudaFree(d_total_pop);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_total_fitness);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_total_pop);
    free(h_total_fitness); 

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}