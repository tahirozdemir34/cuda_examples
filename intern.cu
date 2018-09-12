#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <curand.h>
#include <cuda.h>
#define N 10


__global__
void copyMatPar(float *mat1, float *mat2){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    mat1[index] = mat2[index];
}

__global__
void fill(float *a){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    a[index] = (float) 9999;
}

__global__
void expandRandom(float *a, int min, int max){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    a[index] *= (max - min + 0.9999);
}

__global__
void costCalc(float *a, float * finalCost) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(finalCost, a[index]);
}

__global__ 
void randomExchange(float* array, float *randomFeed){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  atomicExch(&array[blockIdx.x * blockDim.x +( (int)randomFeed[index*2+1] % N )], randomFeed[blockIdx.x * blockDim.x + ( (int)randomFeed[index*2 + 2] %(20*N))  ]);
}
__global__ 
void solver(float *array, int iteration, float *randomFeed, float *solution, float *cost){
   int index = threadIdx.x;
   float *currentSolution = new float[N];
   float *newSolution =new float[N];

   cudaDeviceSynchronize();
   copyMatPar<<<1,N>>>(currentSolution, &array[N*index]);
   copyMatPar<<<1,N>>>(newSolution, &array[N*index]);

   float* minCost = new float;
   cudaDeviceSynchronize();
   for(int i=0; i<iteration; i++){
        randomExchange<<<1,N>>>(newSolution, &randomFeed[(i*i+iteration+123*index)%(20*N)]);
        cudaDeviceSynchronize();
        float *currentCost = new float;
        float *newCost = new float;
        costCalc<<<1,N>>>(newSolution, newCost);
        costCalc<<<1,N>>>(currentSolution, currentCost);
        cudaDeviceSynchronize();
        if(*currentCost>*newCost){
            copyMatPar<<<1,N>>>(currentSolution, newSolution);
            *minCost = *newCost;
        }
   }
   cudaDeviceSynchronize();
   copyMatPar<<<1,N>>>(&solution[N*index], currentSolution);
   cost[index] = *minCost;
   cudaDeviceSynchronize();
}

int main() {
    struct timeval startc, end;
    float ms;
    long seconds, useconds;
    double mtime;

    float *random;
    float *cost;
    float *initialSolutions;
    float *finalSolutions;

    cudaMallocManaged((void**)&initialSolutions, 10*N*sizeof(float));
    cudaMallocManaged((void**)&finalSolutions, 10*N*sizeof(float));
    cudaMallocManaged((void**)&cost, 10*sizeof(float));
    cudaMallocManaged((void **)&random, 20*N*sizeof(float));

    gettimeofday(&startc, NULL);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1235UL);
    curandGenerateUniform(gen, random, 2*10*N);
    expandRandom<<<10,N*2>>>(random, 0, 9999);
    fill<<<10,N>>>(initialSolutions);
    cudaDeviceSynchronize();


    solver<<<1,10>>>(initialSolutions, 100, random, finalSolutions ,cost);
    

    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - startc.tv_sec;
    useconds = end.tv_usec - startc.tv_usec;
    mtime = useconds;
    mtime/=1000;
    mtime+=seconds*1000;
    for(int i=0; i<10; i++){
        for(int j=0; j<N; j++){
            printf("Array[%d,%d]\t%f\t%f\n", i, j,initialSolutions[N*i+j], finalSolutions[N*i+j]);
        }
        printf("---------------\n");
    }
    for(int i=0; i<10; i++){
        printf("cost[%d]: %f\n", i, cost[i]);
    }
    printf("\nTime: %g\n", mtime);


    return 0;
}
