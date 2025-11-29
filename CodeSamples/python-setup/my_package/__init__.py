"""
自定义 CUDA 扩展包
"""
import torch
from torch import Tensor
from typing import Tuple

# 导入编译后的 C++ 扩展
try:
    import my_cuda_ops
except ImportError as e:
    raise ImportError(
        "Failed to import my_cuda_ops. "
        "Please compile the extension first by running: "
        "python setup.py install"
    ) from e

__version__ = "0.1.0"

def add(a: Tensor, b: Tensor) -> Tensor:
    """
    对两个张量执行逐元素加法
    
    Args:
        a: 输入张量 1
        b: 输入张量 2
    
    Returns:
        相加后的张量
    """
    return my_cuda_ops.add_forward(a, b)


def multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    对两个张量执行逐元素乘法
    
    Args:
        a: 输入张量 1
        b: 输入张量 2
    
    Returns:
        相乘后的张量
    """
    return my_cuda_ops.mul_forward(a, b)


def custom_op(input: Tensor, scale: float = 1.0) -> Tuple[Tensor, Tensor]:
    """
    自定义操作示例
    
    Args:
        input: 输入张量
        scale: 缩放因子
    
    Returns:
        (scaled_tensor, relu_tensor) 元组
    """
    return my_cuda_ops.custom_operation(input, scale)


class CustomFunction(torch.autograd.Function):
    """
    自定义 autograd 函数，支持反向传播
    """
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return my_cuda_ops.add_forward(a, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # 加法的梯度就是 grad_output 本身
        return grad_output, grad_output


def custom_add_with_grad(a: Tensor, b: Tensor) -> Tensor:
    """
    支持自动微分的自定义加法
    
    Args:
        a: 输入张量 1
        b: 输入张量 2
    
    Returns:
        相加后的张量
    """
    return CustomFunction.apply(a, b)


__all__ = [
    'add',
    'multiply',
    'custom_op',
    'custom_add_with_grad',
    'CustomFunction',
]