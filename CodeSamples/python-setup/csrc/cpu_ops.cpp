#include <torch/extension.h>

// CPU 版本的加法
torch::Tensor add_cpu(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    TORCH_CHECK(!a.is_cuda(), "add_cpu expects CPU tensors");
    
    return a + b;
}

// CPU 版本的乘法
torch::Tensor mul_cpu(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    TORCH_CHECK(!a.is_cuda(), "mul_cpu expects CPU tensors");
    
    return a * b;
}