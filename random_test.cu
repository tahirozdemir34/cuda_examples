#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <curand.h>
#include <cuda.h>


int main() {
    struct timeval startc, end;
    float ms;
    long seconds, useconds;
    double mtime;
    unsigned long N = 32768;
    float *da, *db;

    cudaProfilerStop();
    gettimeofday(&startc, NULL);
    cudaMallocManaged((void **)&da, N*N*sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234UL);
    curandGenerateUniform(gen, da, N*N);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - startc.tv_sec;
    useconds = end.tv_usec - startc.tv_usec;
    mtime = useconds;
    mtime/=1000;
    mtime+=seconds*1000;
    printf("\nTime of GPU: %g\n", mtime);
    
    gettimeofday(&startc, NULL);
    db = (float *)malloc(N*N*(sizeof(float)));
    for (unsigned long i = 0; i<N*N; i++) {
        db[i] = rand();
    }
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - startc.tv_sec;
    useconds = end.tv_usec - startc.tv_usec;
    mtime = useconds;
    mtime/=1000;
    mtime+=seconds*1000;
    printf("\nTime of CPU.: %g\n", mtime);
    cudaProfilerStop();
    return 0;
}
