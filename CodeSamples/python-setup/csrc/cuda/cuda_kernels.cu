#include <cuda.h>
#include <cuda_runtime.h>

// 加法 kernel
template <typename scalar_t>
__global__ void add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = a[idx] + b[idx];
    }
}

// 乘法 kernel
template <typename scalar_t>
__global__ void mul_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = a[idx] * b[idx];
    }
}

// 显式实例化模板
template __global__ void add_kernel<float>(
    const float*, const float*, float*, const int);
template __global__ void add_kernel<double>(
    const double*, const double*, double*, const int);

template __global__ void mul_kernel<float>(
    const float*, const float*, float*, const int);
template __global__ void mul_kernel<double>(
    const double*, const double*, double*, const int);