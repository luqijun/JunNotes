#include <torch/extension.h>
#include <vector>

// 前向声明 CPU 函数
torch::Tensor add_cpu(torch::Tensor a, torch::Tensor b);
torch::Tensor mul_cpu(torch::Tensor a, torch::Tensor b);

// 前向声明 CUDA 函数
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor mul_cuda(torch::Tensor a, torch::Tensor b);

// 自动选择 CPU 或 CUDA 实现
torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    if (a.is_cuda()) {
        return add_cuda(a, b);
    }
    return add_cpu(a, b);
}

torch::Tensor mul_forward(torch::Tensor a, torch::Tensor b) {
    if (a.is_cuda()) {
        return mul_cuda(a, b);
    }
    return mul_cpu(a, b);
}

// 定义自定义算子
std::vector<torch::Tensor> custom_operation(
    torch::Tensor input,
    float scale) {
    
    auto output1 = input * scale;
    auto output2 = torch::relu(output1);
    
    return {output1, output2};
}

// Python 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PyTorch custom CUDA operations";
    
    m.def("add_forward", &add_forward, "Add two tensors (CPU/CUDA)");
    m.def("mul_forward", &mul_forward, "Multiply two tensors (CPU/CUDA)");
    m.def("custom_operation", &custom_operation, "Custom operation example");
    
    // 可以添加更多函数
    m.def("add_cpu", &add_cpu, "Add two tensors on CPU");
    m.def("add_cuda", &add_cuda, "Add two tensors on CUDA");
}