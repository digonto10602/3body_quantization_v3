#!/bin/bash

#rm printer.o

#nvcc M3_benchmark.cpp -I/usr/include/eigen3/ -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/math_libs/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/math_libs/lib64 -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/lib64 -O3 -std=c++14 -lcublas -lcusolver -lcudart -lgomp -Xcompiler -fopenmp -o printer_gpu

nvcc ./source/Faddeeva.cc -c -o Faddeeva.o -Xcompiler -fopenmp 
nvcc -g -c -o printer_omp.o ./source/test_gpu.cpp -I/usr/include/eigen3/ -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/math_libs/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/math_libs/lib64 -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/lib64 -O3 -std=c++14 -lcublas -lcusolver -lcudart -lgomp -Xcompiler -fopenmp  
nvcc -g -o test_gpu printer_omp.o Faddeeva.o -I/usr/include/eigen3/ -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/math_libs/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/math_libs/lib64 -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/lib64 -O3 -std=c++14 -lcublas -lcusolver -lcudart -lgomp -Xcompiler -fopenmp