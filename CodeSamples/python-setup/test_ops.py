"""
测试自定义 CUDA 扩展
"""
import torch
import my_package
import time

def test_cpu_operations():
    """测试 CPU 操作"""
    print("=" * 50)
    print("Testing CPU Operations")
    print("=" * 50)
    
    a = torch.randn(1000, 1000)
    b = torch.randn(1000, 1000)
    
    # 测试加法
    result_add = my_package.add(a, b)
    expected_add = a + b
    assert torch.allclose(result_add, expected_add), "CPU add failed"
    print("✓ CPU add passed")
    
    # 测试乘法
    result_mul = my_package.multiply(a, b)
    expected_mul = a * b
    assert torch.allclose(result_mul, expected_mul), "CPU multiply failed"
    print("✓ CPU multiply passed")


def test_cuda_operations():
    """测试 CUDA 操作"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return
    
    print("\n" + "=" * 50)
    print("Testing CUDA Operations")
    print("=" * 50)
    
    device = torch.device("cuda")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    
    # 测试加法
    result_add = my_package.add(a, b)
    expected_add = a + b
    assert torch.allclose(result_add, expected_add), "CUDA add failed"
    print("✓ CUDA add passed")
    
    # 测试乘法
    result_mul = my_package.multiply(a, b)
    expected_mul = a * b
    assert torch.allclose(result_mul, expected_mul), "CUDA multiply failed"
    print("✓ CUDA multiply passed")


def test_custom_operation():
    """测试自定义操作"""
    print("\n" + "=" * 50)
    print("Testing Custom Operations")
    print("=" * 50)
    
    x = torch.randn(100, 100)
    scale = 2.5
    
    output1, output2 = my_package.custom_op(x, scale)
    
    expected1 = x * scale
    expected2 = torch.relu(expected1)
    
    assert torch.allclose(output1, expected1), "Custom op output1 failed"
    assert torch.allclose(output2, expected2), "Custom op output2 failed"
    print("✓ Custom operation passed")


def test_autograd():
    """测试自动微分"""
    print("\n" + "=" * 50)
    print("Testing Autograd")
    print("=" * 50)
    
    a = torch.randn(100, 100, requires_grad=True)
    b = torch.randn(100, 100, requires_grad=True)
    
    # 使用自定义 autograd 函数
    c = my_package.custom_add_with_grad(a, b)
    loss = c.sum()
    loss.backward()
    
    # 检查梯度
    assert a.grad is not None, "Gradient for a is None"
    assert b.grad is not None, "Gradient for b is None"
    assert torch.allclose(a.grad, torch.ones_like(a)), "Gradient incorrect for a"
    assert torch.allclose(b.grad, torch.ones_like(b)), "Gradient incorrect for b"
    print("✓ Autograd passed")


def benchmark_operations():
    """性能基准测试"""
    if not torch.cuda.is_available():
        print("\nCUDA not available, skipping benchmark")
        return
    
    print("\n" + "=" * 50)
    print("Performance Benchmark")
    print("=" * 50)
    
    device = torch.device("cuda")
    size = 10000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热
    for _ in range(10):
        _ = my_package.add(a, b)
    torch.cuda.synchronize()
    
    # 自定义操作
    start = time.time()
    for _ in range(100):
        _ = my_package.add(a, b)
    torch.cuda.synchronize()
    custom_time = time.time() - start
    
    # PyTorch 原生操作
    start = time.time()
    for _ in range(100):
        _ = a + b
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    print(f"Custom CUDA operation: {custom_time:.4f} seconds")
    print(f"PyTorch native operation: {pytorch_time:.4f} seconds")
    print(f"Speedup: {pytorch_time/custom_time:.2f}x")


if __name__ == "__main__":
    print("Testing Custom CUDA Extension\n")
    
    try:
        test_cpu_operations()
        test_cuda_operations()
        test_custom_operation()
        test_autograd()
        benchmark_operations()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise