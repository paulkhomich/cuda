#include <stdio.h>

#define N						1000
#define THREADSINBLOCK_ONEDIM	32

__global__ void transp_kernel(int* matrix) {
	int i, j;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j > i && j < N && i < N) {
		int temp = matrix[N*i+j];
		matrix[N*i+j] = matrix[N*j+i];	
		matrix[N*j+i] = temp;
	}
}

int main(void) {
	int matrix[N*N] = {0};	// linear representation of N*N matrix
	for (size_t i = 0; i < N*N; ++i)
		matrix[i] = i

	printf("A[5] = %d\tA[1873] = %d\tA[5000] = %d\tA[872001] = %d\n", matrix[5], matrix[1873], matrix[5000], matrix[872001]);

	int* d_matrix;
	cudaMalloc((void**)&d_matrix, N*N*sizeof(int));
	cudaMemcpy((void*)d_matrix, (void*)matrix, N*N*sizeof(int), cudaMemcpyHostToDevice);

	dim3 block = dim3(THREADSINBLOCK_ONEDIM, THREADSINBLOCK_ONEDIM);
	dim3 grid = dim3((N-1)/THREADSINBLOCK_ONEDIM + 1, (N-1)/THREADSINBLOCK_ONEDIM + 1);
	transp_kernel<<<grid, block>>>(d_matrix);

	cudaMemcpy((void*)matrix, (void*)d_matrix, N*N*sizeof(int), cudaMemcpyDeviceToHost);


	printf("A[5] = %d\tA[1873] = %d\tA[5000] = %d\tA[872001] = %d\n", matrix[5], matrix[1873], matrix[5000], matrix[872001]);
}
