#include <cuda_runtime.h>
#include <stdio.h>

void helloWorld(){
    printf("Hello from the CPU!\n");
}

__global__ void addKernel(int *c, const int *a, const int *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(){
    helloWorld();

    //Size of vectors
    int N = 10;
    size_t size = N * sizeof(int);


    // Allocate host memory
    int h_A[N], h_B[N], h_C[N];
    for(int i = 0; i< N; ++i){
        h_A[i] = i;
        h_B[i] = i*2;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, N);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < N; ++i){
        if (h_C[i] != h_A[i] + h_B[i]){
            success = false;
            printf("Mismatch at index %d: %d != %d\n", i, h_C[i], h_A[i] + h_B[i]);
            break;
        }else{
            printf("Index: %d => %d + %d = %d\n", i, h_A[i], h_B[i], h_C[i]);
        }
    }

    if (success) {
        printf("Vector addition succesful!\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}