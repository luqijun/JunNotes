import numpy as np
import torch
from torch.autograd import Function, gradcheck

print("=" * 70)
print("梯度检查原理详解")
print("=" * 70)

def numerical_gradient(f, x, eps=1e-5):
    """
    使用有限差分法计算数值梯度
    f(x + h) - f(x - h)
    ---------------------
           2h
    """
    grad = torch.zeros_like(x)
    
    for i in range(x.numel()):
        # 创建扰动
        x_plus = x.clone()
        x_minus = x.clone()
        x_plus.view(-1)[i] += eps
        x_minus.view(-1)[i] -= eps
        
        # 计算数值梯度
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        grad.view(-1)[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad

# 定义一个简单函数
def func(x):
    return (x ** 2).sum()

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 解析梯度
y = func(x)
y.backward()
analytical_grad = x.grad.clone()

# 数值梯度
x_no_grad = x.detach().clone()
numerical_grad = numerical_gradient(func, x_no_grad)

print("函数: f(x) = sum(x^2)")
print(f"输入: {x.data}")
print(f"\n解析梯度 (自动微分): {analytical_grad}")
print(f"数值梯度 (有限差分): {numerical_grad}")
print(f"差异: {(analytical_grad - numerical_grad).abs().max().item():.2e}")

print("\n" + "=" * 70)
print("使用 gradcheck 自动验证")
print("=" * 70)

class MyExp(Function):
    @staticmethod
    def forward(ctx, x):
        result = torch.exp(x)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

# 使用 double 精度以提高数值稳定性
x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)

print("测试自定义 Exp 函数...")
test_result = gradcheck(MyExp.apply, x, eps=1e-6, atol=1e-4)
print(f"结果: {'✓ 通过' if test_result else '✗ 失败'}")

print("\n" + "=" * 70)
print("常见梯度错误示例")
print("=" * 70)

class WrongGradient(Function):
    """故意写错的梯度"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # 错误: 应该是 2*x，这里写成 3*x
        return grad_output * 3 * x

print("测试错误的梯度实现...")
x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
try:
    test_result = gradcheck(WrongGradient.apply, x, eps=1e-6, atol=1e-4)
    print(f"结果: {'✓ 通过' if test_result else '✗ 失败'}")
except RuntimeError as e:
    print("✗ 检测到错误: gradcheck 失败")
    print("  说明梯度实现有误!")

print("\n" + "=" * 70)
print("复杂函数的梯度验证")
print("=" * 70)

class ComplexFunction(Function):
    """
    复杂函数: f(x, y) = sin(x) * exp(y) + x^2 * y
    """
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return torch.sin(x) * torch.exp(y) + x**2 * y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        
        # df/dx = cos(x)*exp(y) + 2*x*y
        grad_x = grad_output * (torch.cos(x) * torch.exp(y) + 2 * x * y)
        
        # df/dy = sin(x)*exp(y) + x^2
        grad_y = grad_output * (torch.sin(x) * torch.exp(y) + x**2)
        
        return grad_x, grad_y

x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
y = torch.randn(2, 3, dtype=torch.double, requires_grad=True)

print("测试复杂函数...")
test_result = gradcheck(ComplexFunction.apply, (x, y), eps=1e-6, atol=1e-4)
print(f"结果: {'✓ 通过' if test_result else '✗ 失败'}")

print("\n" + "=" * 70)
print("梯度检查最佳实践")
print("=" * 70)
print("1. 使用 double 精度 (torch.double)")
print("2. 设置合适的 eps (通常 1e-6)")
print("3. 设置合适的 atol (绝对容差，通常 1e-4)")
print("4. 测试多个随机输入")
print("5. 在开发阶段测试，生产环境可以关闭")
print("=" * 70)