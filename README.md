## This repository contains a few CUDA example. 

* random_test.cu: A simple comparison for CPU and GPU by creating a large size array with random values. 
* matrix_mul.cu: Square matrix multiplication up to 1024x1024. Also, contains CPU-GPU comparison.
* sim_test.cu: Simulated Annealing algorithm with CUDA. Not the best approach for writing such an algorithm but it performs better for some parameters. Also, CPU-GPU comparison is included.
* intern.cu: A simple program that is written for my internship. Main aim was practicing for cuRand library. 

## Sample build instruction

nvcc sim_test.cu -o sim_test.out -lcurand -arch compute_35 -lcudadevrt -rdc=true --default-stream per-thread


[CUDA İçin Temel Bilgiler](https://medium.com/@tahirozdemir34/cuda-i%CC%87%C3%A7in-temel-bilgiler-d22e038212f1)
