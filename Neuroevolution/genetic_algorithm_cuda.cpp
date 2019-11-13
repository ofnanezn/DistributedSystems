%%cu
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


__global__ void individual_evolution(float *total_pop, float *total_fitness, float *new_pop, 
                                float *child1, float *child2, int dims, int pop_size){
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < pop_size * dims){
    int p1, p2;

    curandState state;
    curand_init(clock64(), i, 0, &state);
        
    int parent2_pos = (int)(curand_uniform(&state) * pop_size);
    
    // Xover
    int crosspoint = (int)(curand_uniform(&state)*(dims - 1) + 1);
    for(int j = 0; j < dims; ++j){
      p1 = i * dims + j;
      p2 = parent2_pos * dims + j;
      child1[j] = j < crosspoint ? total_pop[p1] : total_pop[p2];
      child2[j] = j < crosspoint ? total_pop[p2] : total_pop[p1];
    }

    //Mutation
    int mut1_pos = (int)(curand_uniform(&state) * dims);
    int mut2_pos = (int)(curand_uniform(&state) * dims);

    child1[mut1_pos] += curand_uniform(&state) * 0.1;
    child2[mut2_pos] += curand_uniform(&state) * 0.1;

    float ch1_fitness = 10.0 * dims;
    float ch2_fitness = 10.0 * dims;

    for(int j = 0; j < dims; ++j){
      ch1_fitness += (child1[j]*child1[j] - 10 * cos(2.0 * PI * child1[j]));
      ch2_fitness += (child2[j]*child2[j] - 10 * cos(2.0 * PI * child2[j]));       
    }

    int k;
    if(ch1_fitness <= total_fitness[i] && ch1_fitness <= ch2_fitness){
      total_fitness[i] = ch1_fitness;
      for(int j = 0; j < dims; ++j){
        k = i * dims + j;
        new_pop[k] = child1[j];
      }          
    }
    else if(ch2_fitness <= total_fitness[i] && ch2_fitness < ch1_fitness){
      total_fitness[i] = ch2_fitness;
      for(int j = 0; j < dims; ++j){
        k = i * dims + j;
        new_pop[k] = child2[j];
      }          
    }
    else{
      for(int j = 0; j < dims; ++j){
        k = i * dims + j;
        new_pop[k] = total_pop[k];
      }
    }
  } 
}


/*******************************************************************************/
int main(void){
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  int T = 50;
  int dims = 10, pop_size = 10000;

  /******************************** 
  Population initialization
  *********************************/
  // Print the vector length to be used, and compute its size
  int total_pop_size = dims * pop_size;
  size_t total_size = total_pop_size * sizeof(float);
  printf("[Island Genetic Algorithm with %d elements]\n", total_pop_size);

  float *h_total_pop = (float *)malloc(total_size);
  
  if(h_total_pop == NULL){
      fprintf(stderr, "Failed to allocate host total_pop!\n");
      exit(EXIT_FAILURE);
  }

  float *d_total_pop = NULL; 
  err = cudaMalloc((void **) &d_total_pop, total_size);

  if (err != cudaSuccess){
      fprintf(stderr, "Failed to allocate device vector total_pop (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  printf("Copy total_pop from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_total_pop, h_total_pop, total_size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess){
      fprintf(stderr, "Failed to copy total_pop from host to device!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  int threadsPerBlock = 128;
  int blocksPerGrid =(total_pop_size + threadsPerBlock - 1) / threadsPerBlock;

  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  initialize_pop<<<blocksPerGrid, threadsPerBlock>>>(d_total_pop, total_pop_size);

  err = cudaGetLastError();

  if (err != cudaSuccess){
      fprintf(stderr, "Failed to launch initialize_pop kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_total_pop, d_total_pop, total_size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess){
      fprintf(stderr, "Failed to copy total_pop from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  /******************************** 
  Fitness evaluation
  *********************************/
  int fitness_elements = pop_size;
  size_t total_fitness_size = fitness_elements * sizeof(float);

  float *h_total_fitness = (float *)malloc(total_fitness_size);

  if (h_total_fitness == NULL){
    fprintf(stderr, "Failed to allocate host total_fitness!\n");
    exit(EXIT_FAILURE);
  }

  float *d_total_fitness = NULL; 
  err = cudaMalloc((void **) &d_total_fitness, total_fitness_size);

  if (err != cudaSuccess){
      fprintf(stderr, "Failed to allocate device total_fitness (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  printf("Copy total_fitness from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_total_fitness, h_total_fitness, total_fitness_size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess){
      fprintf(stderr, "Failed to copy total_fitness from host to device!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  threadsPerBlock = 128;
  blocksPerGrid = (fitness_elements + threadsPerBlock - 1) / threadsPerBlock;

  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  calculate_fitness<<<blocksPerGrid, threadsPerBlock>>>(d_total_pop, d_total_fitness, dims, fitness_elements);

  err = cudaGetLastError();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to launch calculate_fitness kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy total_fitness from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_total_fitness, d_total_fitness, total_fitness_size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy total_fitness from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *child1 = NULL; 
  err = cudaMalloc((void **) &child1, dims * sizeof(float));

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device child1 (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *child2 = NULL; 
  err = cudaMalloc((void **) &child2, dims * sizeof(float));

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device child2 (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *h_new_pop = (float *)malloc(total_size);

  if(h_new_pop == NULL){
      fprintf(stderr, "Failed to allocate host new_pop!\n");
      exit(EXIT_FAILURE);
  }

  float *d_new_pop = NULL; 
  err = cudaMalloc((void **) &d_new_pop, total_size);

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device new_pop (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }    

  threadsPerBlock = 128;
  blocksPerGrid = (fitness_elements + threadsPerBlock - 1) / threadsPerBlock;

  for(int t = 0; t < T; ++t){
    err = cudaMemcpy(d_total_pop, h_total_pop, total_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess){
      fprintf(stderr, "Failed to copy total_pop from host to device (error code %s) at iteration %d!\n", cudaGetErrorString(err), t);
      exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_total_fitness, h_total_fitness, total_fitness_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
      fprintf(stderr, "Failed to copy total_fitness from host to device (error code %s) at iteration %d!\n", cudaGetErrorString(err), t);
      exit(EXIT_FAILURE);
    }

    individual_evolution<<<blocksPerGrid, threadsPerBlock>>>(d_total_pop, d_total_fitness, d_new_pop, 
                                                            child1, child2, dims, pop_size);
                             
    err = cudaGetLastError();

    if (err != cudaSuccess){
      fprintf(stderr, "Failed to launch individual_evolution kernel (error code %s) at iteration %d!\n", cudaGetErrorString(err), t);
      exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_total_pop, d_total_pop, total_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
      fprintf(stderr, "Failed to copy total_pop from device to host (error code %s) at iteration %d!\n", cudaGetErrorString(err), t);
      exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_total_fitness, d_total_fitness, total_fitness_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
      fprintf(stderr, "Failed to copy total_fitness from device to host (error code %s) at iteration %d!\n", cudaGetErrorString(err), t);
      exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_new_pop, d_new_pop, total_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
      fprintf(stderr, "Failed to copy new_pop from device to host (error code %s) at iteration %d!\n", cudaGetErrorString(err), t);
      exit(EXIT_FAILURE);
    }

    h_total_pop = h_new_pop;
  }

  for (int i = 0; i < 1000; ++i)
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

  err = cudaFree(d_new_pop);

  if (err != cudaSuccess){
      fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaFree(child1);

  if (err != cudaSuccess){
      fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaFree(child2);

  if (err != cudaSuccess){
      fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_total_pop);
  free(h_total_fitness);
  free(h_new_pop); 

  // Reset the device and exit
  err = cudaDeviceReset();

  if (err != cudaSuccess){
      fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  printf("Done\n");
  return 0;
}