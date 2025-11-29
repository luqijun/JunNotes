#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel 的前向声明
template <typename scalar_t>
__global__ void add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ output,
    const int size);

template <typename scalar_t>
__global__ void mul_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ output,
    const int size);

// CUDA 版本的加法
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "add_cuda expects CUDA tensors");
    TORCH_CHECK(b.is_cuda(), "add_cuda expects CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    
    auto output = torch::empty_like(a);
    const int size = a.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "add_cuda", ([&] {
        add_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));
    
    return output;
}

// CUDA 版本的乘法
torch::Tensor mul_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "mul_cuda expects CUDA tensors");
    TORCH_CHECK(b.is_cuda(), "mul_cuda expects CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    
    auto output = torch::empty_like(a);
    const int size = a.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "mul_cuda", ([&] {
        mul_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));
    
    return output;
}