#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  torch::Tensor tensor_cuda = torch::rand({2, 3}).to(torch::kCUDA);
  std::cout << tensor_cuda << std::endl;
}