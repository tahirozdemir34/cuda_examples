#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <curand.h>
#include <cuda.h>
#define N 1024

__global__
void add(int row, int column,  float *a, float *b, float *c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sum;
    sum = 0;
    __syncthreads();
    atomicAdd(&sum, a[index + row * N] * b[index * N + column]);
    __syncthreads();
    c[row*N + column] = sum;
}

__global__
void multiply(float *a, float* b, float* c) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if((row) < N && (column) < N){
        add<<<1,N>>>(row, column, a, b ,c);
    cudaDeviceSynchronize();
    }
}

int main() {
    struct timeval startc, end;
    float ms;
    long seconds, useconds;
    double mtime;
    float *a, *b, *c;

    cudaMallocManaged((void **)&a, N*N*sizeof(float));
    cudaMallocManaged((void **)&b, N*N*sizeof(float));
    cudaMallocManaged((void **)&c, N*N*sizeof(float));


    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234UL);
    curandGenerateUniform(gen, a, N*N);
    curandGenerateUniform(gen, b, N*N);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock(N,N);
    dim3 blocksPerGrid(1, 1);
    if(N*N > 1024){
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        blocksPerGrid.x = ceil((double)N/(double)32);
        blocksPerGrid.y = ceil((double)N/(double)32);
	printf("%d - %d , %d - %d\n", blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
    }
    gettimeofday(&startc, NULL);
    multiply<<<blocksPerGrid, threadsPerBlock>>>(a, b, c);
    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - startc.tv_sec;
    useconds = end.tv_usec - startc.tv_usec;
    mtime = useconds;
    mtime/=1000;
    mtime+=seconds*1000;
    printf("\nGPU Time: %g\n", mtime);

    float *hostC = (float *)malloc(N*N*(sizeof(float)));
    gettimeofday(&startc, NULL);
    for(int i = 0; i< N; i++){
        for(int j = 0; j < N; j++){
            hostC[i*N+j] = 0;
            for(int k = 0; k < N; k++){
                hostC[i*N+j] += a[i*N+k] * b[k*N+j]; 
            }
        }
    }
    gettimeofday(&end, NULL);
    free(hostC);
    seconds  = end.tv_sec  - startc.tv_sec;
    useconds = end.tv_usec - startc.tv_usec;
    mtime = useconds;
    mtime/=1000;
    mtime+=seconds*1000;
    printf("CPU Time: %g\n", mtime);
    return 0;
}
