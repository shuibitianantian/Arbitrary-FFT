#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <random>
#include <iostream>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include "device_launch_parameters.h"

#define pi 3.1415926535
// #define LENGTH 1024*1024 //signal sampling points

std::mt19937_64 R(std::random_device{}());
double randReal(double lo, double up) {
    return std::uniform_real_distribution<double>(lo, up)(R);
}

int main(int argc, char* argv[])
{
    // data gen
    int LENGTH = atoi(argv[1]);
    double* Data = (double*) malloc(LENGTH*sizeof(double));
    for(int i = 0; i < LENGTH; ++i){
        Data[i] = randReal(-100, 100);
    }
    
    double fs = 1000000.000;//sampling frequency
    double f0 = 200000.00;// signal frequency

    cufftComplex *CompData = (cufftComplex*)malloc(LENGTH * sizeof(cufftComplex));//allocate memory for the data in host
    int i;
    for (i = 0; i < LENGTH; i++)
    {
        CompData[i].x = Data[i];
        CompData[i].y = 0;
    }

    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData, LENGTH * sizeof(cufftComplex));// allocate memory for the data in device
    cudaMemcpy(d_fftData, CompData, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);// copy data from host to device

    cufftHandle plan;// cuda library function handle
    cufftPlan1d(&plan, LENGTH, CUFFT_C2C, 1);//declaration
    double tt = omp_get_wtime();
    cufftExecC2C(plan, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done
    cudaMemcpy(CompData, d_fftData, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);// copy the result from device to host

    std::cout << omp_get_wtime() - tt << std::endl;

    cufftDestroy(plan);
    free(CompData);
    cudaFree(d_fftData);

}