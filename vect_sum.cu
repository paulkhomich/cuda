#include <stdio.h>

#define SIZE 100000
#define THREADSINBLOCK 512

__global__ void sum_kernel(int* a, int* b, int* c, int size) {
	int i = blockIdx.x*blockDim.x + threadIdx.x; // thread index in all grid
	if (i < size)
		c[i] = a[i] + b[i];
}

int main(void) {
	int a[SIZE];
	int b[SIZE];
	int c[SIZE];
	for (size_t i = 0; i < SIZE; ++i) {
		a[i] = i%THREADSINBLOCK;
		b[i] = i*SIZE%THREADSINBLOCK;
	}


	int* a_device = 0;
	int* b_device = 0;
	int* c_device = 0;
	cudaMalloc((void**)&a_device, SIZE * sizeof(int));
	cudaMalloc((void**)&b_device, SIZE * sizeof(int));
	cudaMalloc((void**)&c_device, SIZE * sizeof(int));

	cudaMemcpy((void*)a_device, (void*)a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)b_device, (void*)b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	dim3 blockDim = (THREADSINBLOCK);
	dim3 gridDim = ((SIZE - 1)/THREADSINBLOCK  + 1);
	sum_kernel<<<gridDim, blockDim>>>(a_device, b_device, c_device, SIZE);
	
	cudaMemcpy((void*)c, (void*)c_device, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	printf("A:\t%d\t%d\t%d\n", a[0], a[1], a[2]);
	printf("B:\t%d\t%d\t%d\n", b[0], b[1], b[2]);
	printf("C:\t%d\t%d\t%d\n", c[0], c[1], c[2]);
}
