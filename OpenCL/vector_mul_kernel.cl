__kernel void vector_mul(__global const int *A, __global const int *B, __global int *C, int matrix_size) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
    
 
    // Do the operation
    unsigned int cell_sum = 0;
    int row = (i/matrix_size) * matrix_size;
    int col = (i%matrix_size);

    for( unsigned int k = 0; k < matrix_size; k++){
    	cell_sum += A[row + k] * B[col+(k*matrix_size)];
    }    

    C[i] = cell_sum;
}