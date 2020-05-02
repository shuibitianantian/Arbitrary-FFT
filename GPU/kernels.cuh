#include "utils.cuh"

// self defined complex exponential cuda
__device__ cuDoubleComplex cuda_complex_exp (cuDoubleComplex arg){
    double s, c;
    double e = exp(cuCreal(arg));
    sincos(cuCimag(arg), &s, &c);
    return make_cuDoubleComplex(c * e, s * e);
}

// check if there is an error happened in gpu
void Check_CUDA_Error(const char *message){
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
        exit(-1);
    }
}

// gpu dft kernel using 2d grid
__global__ void dft_kernel2d(cuDoubleComplex* X, cuDoubleComplex* Y, const std::size_t N, std::size_t col_size){
  __shared__ cuDoubleComplex cache[BLOCKSIZE];
  
  std::size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
  std::size_t idy = threadIdx.y + blockIdx.y*blockDim.y;

  if(idx < N && idy < N){
    
    cuDoubleComplex tmp = cuda_complex_exp(make_cuDoubleComplex(0, -_2Pi / (Real) N * (Real) idx * (Real) idy));
    cache[threadIdx.x * blockDim.y + threadIdx.y] = cuCmul(tmp, X[idy]);
    __syncthreads();
    // perform reduction in block
    for (unsigned int s = blockDim.y/2; s>0; s>>=1) {
        if (threadIdx.y < s) {
            cache[threadIdx.x * blockDim.y + threadIdx.y] = cuCadd(cache[threadIdx.x * blockDim.y + threadIdx.y], cache[threadIdx.x * blockDim.y + threadIdx.y + s]);
        }
        __syncthreads();
    }

    if(threadIdx.y == 0)
      Y[idx*col_size + blockIdx.y] = cache[threadIdx.x * blockDim.y];
  }
}

__global__ void idft_kernel2d(cuDoubleComplex* X, cuDoubleComplex* Y, const std::size_t N, std::size_t col_size){
  __shared__ cuDoubleComplex cache[BLOCKSIZE];
  
  std::size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
  std::size_t idy = threadIdx.y + blockIdx.y*blockDim.y;

  if(idx < N && idy < N){
    
    cuDoubleComplex tmp = cuda_complex_exp(make_cuDoubleComplex(0, _2Pi / (Real) N * (Real) idx * (Real) idy));
    cache[threadIdx.x * blockDim.y + threadIdx.y] = cuCmul(tmp, X[idy]);
    __syncthreads();
    // perform reduction in block
    for (unsigned int s = blockDim.y/2; s>0; s>>=1) {
        if (threadIdx.y < s) {
            cache[threadIdx.x * blockDim.y + threadIdx.y] = cuCadd(cache[threadIdx.x * blockDim.y + threadIdx.y], cache[threadIdx.x * blockDim.y + threadIdx.y + s]);
        }
        __syncthreads();
    }

    if(threadIdx.y == 0)
      Y[idx*col_size + blockIdx.y] = cache[threadIdx.x * blockDim.y];
  }
}

// perform reduction on row of a 2d matrix
__global__ void reduction_2d(cuDoubleComplex* Y, std::size_t N, std::size_t col_size){
  std::size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
  cuDoubleComplex tmp = make_cuDoubleComplex(0.0, 0.0);
  if(idx < N){
    for(int i = 0; i < col_size; ++i){
      tmp = cuCadd(tmp, Y[idx*col_size + i]);
    }
    Y[idx*col_size] = tmp;
  }
}