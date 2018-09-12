#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <math.h>
#define N 100
#define PAR_SIM_COUNT 5
#define T 1
#define T_MIN 0.1
#define ALPHA 0.9
#define ITERATION 1000000

__global__
void expandRandom(float *a, int min, int max){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	a[index] *= (max - min + 0.9999);
}

__device__
int expandRandom(float a, int min, int max){
	return (int) ((a - 1) * (min - max) / 1 + min);
}

__global__
void fill(int *a){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	a[index] = index%N;
}

__global__
void produceInitial(curandState* globalState,int *a){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = globalState[index];
	for(int i =0; i<N; i++){
		
		int rnd1 = expandRandom(curand_uniform( &localState ),0,N-1);
		int rnd2 = expandRandom(curand_uniform( &localState ),0,N-1);
		int temp = a[index*N+rnd1];
		a[index*N+rnd1] = a[index*N+rnd2];
		a[index*N+rnd2] = temp;
	}
}

__global__
void costCalc(int *a, int *costMatrix, int* calculatedCost) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int sum;
	sum = 0;
	__syncthreads();
	atomicAdd(&sum, costMatrix[a[index]*N + a[index+1]]);
	__syncthreads();
	*calculatedCost = sum;
	
}


__global__
void copyMatPar(int *mat1, int *mat2, int ind){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	mat1[N*ind + index] = mat2[index];
}

__global__
void initSolCopy(int *mat1, int *init, int ind){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	mat1[index] = init[N*ind + index];
}

__global__
void setup_kernel ( curandState * state, unsigned long seed )
{
	int id = threadIdx.x;
	curand_init ( seed, id, 0, &state[id] );
} 

__global__ 
void solver( curandState* globalState, int* init_solution, int * costMatrix, int *solution, int *cost, int* initCosts) 
{

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("Thred %d: Init cost: %d\n", ind, initCosts[ind]);
	float temperature = T;
	curandState localState = globalState[ind];
	int *currentSol = new int[N];
	int *newSol = new int[N];
	initSolCopy<<<1,N>>>(currentSol, init_solution, ind);
	initSolCopy<<<1,N>>>(newSol, init_solution, ind);

	int *currentCost = new int;
	*currentCost = initCosts[ind];
	int * newCost = new int;
	while(temperature > T_MIN){
		for(int i=0; i<ITERATION/500; i++){
			int rnd1 = expandRandom(curand_uniform( &localState ),0,N-1);
			int rnd2 = expandRandom(curand_uniform( &localState ),0,N-1);
			if(rnd1 == rnd2) 
				continue;
			int temp = newSol[rnd1];
			newSol[rnd1] = newSol[rnd2];
			newSol[rnd2] = temp;
			*newCost = 0;
			*currentCost = 0;
			costCalc<<<1,N>>>(currentSol, costMatrix, currentCost);
			costCalc<<<1,N>>>(newSol, costMatrix, newCost);
			cudaDeviceSynchronize();
			if(*newCost < *currentCost){
				*currentCost = *newCost;
				currentSol[rnd1] = newSol[rnd1];
				currentSol[rnd2] = newSol[rnd2];
			
			}
			else if(curand_uniform( &localState ) < exp(((*currentCost- *newCost)/temperature))){
				*currentCost = *newCost;
				currentSol[rnd1] = newSol[rnd1];
				currentSol[rnd2] = newSol[rnd2];
			}
			else{
				newSol[rnd1] = currentSol[rnd1];
				newSol[rnd2] = currentSol[rnd2];
			}
			
		}
		temperature *= ALPHA;
	}

	copyMatPar<<<1,N>>>(solution, currentSol, ind);
	cost[ind] = *currentCost;
}


int main() {
	struct timeval startc, end;

	long seconds, useconds;
	double mtime;
	int *cost_matrix;
	int* init_sol;
	int *finalSolutions;
	int *finalCosts;
	curandState* devStates;
	cudaMalloc(&devStates, N*sizeof(curandState));
	printf("Started\n");
	cudaMallocManaged((void **)&cost_matrix, N*N*sizeof(int));
	cudaMallocManaged((void **)&finalSolutions, N*N*sizeof(int));
	cudaMallocManaged((void **)&finalCosts, N*sizeof(int));
	cudaMallocManaged((void **)&init_sol, (PAR_SIM_COUNT+1)*N*sizeof(int));
	srand(time(0));
	for(int i = 0; i<N*N; i++){
		cost_matrix[i] = rand()%100;
	} 		
	/*for(int i = 0; i<N; i++){
		for(int j = 0; j<N; j++){
			printf("%f\t",cost_matrix[i*N+j]);
		} 
		printf("\n");
	} 		*/
	fill<<<PAR_SIM_COUNT+1,N>>>(init_sol);
	setup_kernel <<<1, N>>>(devStates, time(NULL));
	cudaDeviceSynchronize();
	produceInitial<<<1,PAR_SIM_COUNT+1>>>(devStates,init_sol);
	int *init_costs;
	cudaMallocManaged((void **)&init_costs, N*sizeof(int));
	for(int i = 0; i<PAR_SIM_COUNT+1; i++)
		costCalc<<<1,N>>>(&init_sol[i*N], cost_matrix, &init_costs[i]);	
	cudaDeviceSynchronize();
  gettimeofday(&startc, NULL);
	solver<<<1, PAR_SIM_COUNT>>> (devStates, init_sol, cost_matrix, finalSolutions, finalCosts, init_costs);

	cudaDeviceSynchronize();

	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - startc.tv_sec;
	useconds = end.tv_usec - startc.tv_usec;
	mtime = useconds;
	mtime/=1000;
	mtime+=seconds*1000;
	/*for(int i = 0; i<PAR_SIM_COUNT; i++){
		printf("Init Cost %d: %d\n", i, init_costs[i]);
	}*/
	printf("GPU Solution: ");
	printf("\nTime of GPU: %g\n", mtime);
	//See all solutions and its costs
	/*for(int i=0; i<PAR_SIM_COUNT; i++){
		for(int j=0; j<N; j++){
			printf("Array[%d,%d]\t%d\n", i, j, finalSolutions[N*i+j]);
		}
		printf("---------------\n");
	}
	for(int i=0; i<PAR_SIM_COUNT; i++){
		printf("cost[%d]: %d\n", i, finalCosts[i]);
	}*/

	int minCost = finalCosts[0];
	//int minCostIndex = 0;

	for(int i=1; i<PAR_SIM_COUNT; i++){
		if(minCost > finalCosts[i]){
			minCost = finalCosts[i];
			//minCostIndex = i;
		}
	}

	/*for(int i = 0; i<N; i++){
		printf("%d -> ", finalSolutions[N*minCostIndex + i]);
	}*/
	printf("GPU Cost: %d\n\n",minCost);
	
	//CPU test
	{
		srand(time(0));
		int currentSol[N];
		int newSol[N];
		gettimeofday(&startc, NULL);
		
		float temperature = T;
		float alpha = ALPHA;
		float t_min = T_MIN;
		int currentCost = init_costs[PAR_SIM_COUNT];


 		for(int i = 0; i<N; i++){
 			currentSol[i] = init_sol[PAR_SIM_COUNT*N+i];
 			newSol[i] = init_sol[PAR_SIM_COUNT*N+i];
		}

		int newCost = 0;
		printf("\nCPU Init: %d\n", currentCost);
		while(temperature > t_min){
			for(int i=0; i<ITERATION; i++){
				int rnd1 = rand()%N;
				int rnd2 = rand()%N;
				if(rnd1 == rnd2) 
					continue;
				int temp = newSol[rnd1];
				newSol[rnd1] = newSol[rnd2];
				newSol[rnd2] = temp;
				currentCost =0;
	      newCost =0;
				for(int i= 0; i<N-1; i++){
					currentCost += cost_matrix[currentSol[i]*N+currentSol[i+1]];
					newCost += cost_matrix[newSol[i]*N+newSol[i+1]];
				}
				if(newCost < currentCost){
					currentCost = newCost;
					currentSol[rnd1] = newSol[rnd1];
					currentSol[rnd2] = newSol[rnd2];
			
				}
				else if(((double)rand() / (double)RAND_MAX)< exp(((currentCost -newCost )/temperature))){
					currentCost = newCost;
					currentSol[rnd1] = newSol[rnd1];
					currentSol[rnd2] = newSol[rnd2];
				}
				else{
					newSol[rnd1] = currentSol[rnd1];
					newSol[rnd2] = currentSol[rnd2];
				}
			}
			temperature *= alpha;
		}

		printf("CPU Solution: ");
		/*for(int i = 0; i<N; i++){
			printf("%d -> ", currentSol[i]);
		}*/

		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - startc.tv_sec;
		useconds = end.tv_usec - startc.tv_usec;
		mtime = useconds;
		mtime/=1000;
		mtime+=seconds*1000;
		printf("\nTime of CPU: %g\n", mtime);
		printf("CPU Cost: %d", currentCost);

	}

	printf("\n");
	return 0;
}
