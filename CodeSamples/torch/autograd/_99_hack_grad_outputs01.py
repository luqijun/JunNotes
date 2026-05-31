import torch
import torch.nn as nn

print("=" * 70)
print("理解 grad_outputs: 从链式法则开始")
print("=" * 70)

print("""
链式法则回顾:
  如果 z = f(y), y = g(x), 那么:
  dz/dx = dz/dy * dy/dx
  
在 PyTorch 中:
  - dy/dx 是我们要计算的
  - dz/dy 就是 grad_outputs (从后续计算传来的梯度)
""")

print("\n" + "=" * 70)
print("示例 1: 标量输出 - grad_outputs 默认为 1")
print("=" * 70)

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x**2).sum()  # y 是标量

print(f"x = {x.data}")
print(f"y = sum(x^2) = {y.item()}")

# 对于标量，backward() 不需要参数
# 相当于 y.backward(torch.tensor(1.0))
y.backward()

print(f"\nx.grad = {x.grad}")
print("解释: dy/dx = 2*x = [4.0, 6.0]")
print("      相当于 dy/dy * dy/dx = 1.0 * [4.0, 6.0]")

print("\n" + "=" * 70)
print("示例 2: 向量输出 - 必须指定 grad_outputs")
print("=" * 70)

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x**2  # y 是向量 [4, 9]

print(f"x = {x.data}")
print(f"y = x^2 = {y.data}")

# 向量输出必须指定 grad_outputs
grad_outputs = torch.tensor([1.0, 1.0])
y.backward(grad_outputs)

print(f"\ngrad_outputs = {grad_outputs}")
print(f"x.grad = {x.grad}")
print("\n计算过程:")
print("  dy[0]/dx[0] = 2*x[0] = 4")
print("  dy[1]/dx[1] = 2*x[1] = 6")
print("  最终梯度 = grad_outputs * dy/dx")
print("           = [1.0, 1.0] * [4.0, 6.0]")
print("           = [4.0, 6.0]")

print("\n" + "=" * 70)
print("示例 3: grad_outputs 的加权作用")
print("=" * 70)

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x**2

# 使用不同的权重
grad_outputs = torch.tensor([0.5, 2.0])
y.backward(grad_outputs)

print(f"x = {x.data}")
print(f"y = x^2 = {y.data}")
print(f"\ngrad_outputs = {grad_outputs}")
print(f"x.grad = {x.grad}")
print("\n计算过程:")
print("  x.grad[0] = grad_outputs[0] * dy[0]/dx[0]")
print("            = 0.5 * 4.0 = 2.0")
print("  x.grad[1] = grad_outputs[1] * dy[1]/dx[1]")
print("            = 2.0 * 6.0 = 12.0")

print("\n" + "=" * 70)
print("示例 4: 理解为什么标量不需要 grad_outputs")
print("=" * 70)

x = torch.tensor([2.0, 3.0], requires_grad=True)

# 方式1: 向量输出求和得到标量
y_vector = x**2
y_scalar = y_vector.sum()

print("方式1: 先平方再求和")
y_scalar.backward()
grad1 = x.grad.clone()
print(f"  x.grad = {grad1}")

# 方式2: 向量输出 + grad_outputs=[1,1]
x.grad = None
y_vector = x**2
y_vector.backward(torch.ones_like(y_vector))
grad2 = x.grad.clone()

print("\n方式2: 向量输出 + grad_outputs=[1,1]")
print(f"  x.grad = {grad2}")

print(f"\n两种方式结果相同: {torch.allclose(grad1, grad2)}")
print("\n结论: sum() 操作隐式地使用了 grad_outputs=[1,1,...]")

print("\n" + "=" * 70)
print("示例 5: 多输出情况")
print("=" * 70)

x = torch.tensor([2.0], requires_grad=True)

# 计算多个输出
y1 = x**2
y2 = x**3

print(f"x = {x.item()}")
print(f"y1 = x^2 = {y1.item()}")
print(f"y2 = x^3 = {y2.item()}")

# 场景: 想要计算 z = 2*y1 + 3*y2 对 x 的梯度
# 方法1: 直接计算
z = 2 * y1 + 3 * y2
z.backward()
print(f"\n方法1 (直接): x.grad = {x.grad.item()}")

# 方法2: 分别计算后组合
x.grad = None
y1 = x**2
y1.backward(retain_graph=True)
grad_from_y1 = x.grad.clone() * 2  # dy1/dx * 2

x.grad = None
y2 = x**3
y2.backward()
grad_from_y2 = x.grad * 3  # dy2/dx * 3

total_grad = grad_from_y1 + grad_from_y2
print(f"方法2 (组合): x.grad = {total_grad.item()}")

print("\n计算过程:")
print("  dz/dx = dz/dy1 * dy1/dx + dz/dy2 * dy2/dx")
print("        = 2 * (2*x) + 3 * (3*x^2)")
print(f"        = 2 * 4 + 3 * 12 = {2 * 4 + 3 * 12}")

print("\n" + "=" * 70)
print("示例 6: 计算雅可比矩阵 - grad_outputs 的核心应用")
print("=" * 70)


def compute_jacobian(func, x):
    """
    计算函数的雅可比矩阵
    func: R^n -> R^m
    Jacobian[i,j] = ∂y_i/∂x_j
    """
    x = x.detach().requires_grad_(True)
    y = func(x)

    m, n = y.shape[0], x.shape[0]
    jacobian = torch.zeros(m, n)

    for i in range(m):
        # 创建第 i 个单位向量作为 grad_outputs
        grad_output = torch.zeros_like(y)
        grad_output[i] = 1.0

        if x.grad is not None:
            x.grad.zero_()

        y = func(x)
        y.backward(grad_output, retain_graph=True)

        # 保存第 i 行
        jacobian[i] = x.grad

    return jacobian


# 定义向量函数
def vector_func(x):
    """
    f: R^2 -> R^3
    y1 = x1^2 + x2
    y2 = x1 * x2
    y3 = x1 + x2^2
    """
    return torch.tensor([x[0] ** 2 + x[1], x[0] * x[1], x[0] + x[1] ** 2])


x = torch.tensor([2.0, 3.0])
jacobian = compute_jacobian(vector_func, x)

print(f"输入: x = {x}")
print(f"输出: y = {vector_func(x)}")
print("\n雅可比矩阵:")
print(jacobian)

print("\n理论值:")
print("  ∂y1/∂x1 = 2*x1 = 4,   ∂y1/∂x2 = 1")
print("  ∂y2/∂x1 = x2 = 3,     ∂y2/∂x2 = x1 = 2")
print("  ∂y3/∂x1 = 1,          ∂y3/∂x2 = 2*x2 = 6")

print("\ngrad_outputs 的作用:")
print("  grad_outputs = [1, 0, 0] -> 计算雅可比矩阵第1行")
print("  grad_outputs = [0, 1, 0] -> 计算雅可比矩阵第2行")
print("  grad_outputs = [0, 0, 1] -> 计算雅可比矩阵第3行")

print("\n" + "=" * 70)
print("示例 7: 向量-雅可比积 (VJP)")
print("=" * 70)


def func(x):
    return torch.stack([x[0] ** 2, x[0] * x[1], x[1] ** 2])


x = torch.tensor([2.0, 3.0], requires_grad=True)
v = torch.tensor([1.0, 2.0, 3.0])  # 左乘向量

y = func(x)
# 计算 v^T @ J(x)
y.backward(v)

print(f"x = {x.data}")
print(f"y = f(x) = {y.data}")
print(f"v = {v}")
print(f"\nv^T @ J(x) = {x.grad}")

print("\n手动计算验证:")
print("J(x) = [[2*x[0],    0   ],")
print("        [x[1],      x[0] ],")
print("        [0,         2*x[1]]]")
print(f"     = [[{2 * x[0].item():.1f}, {0:.1f}],")
print(f"        [{x[1].item():.1f}, {x[0].item():.1f}],")
print(f"        [{0:.1f}, {2 * x[1].item():.1f}]]")
print(f"\nv^T @ J = [{1 * 2 * x[0].item() + 2 * x[1].item():.1f}, {2 * x[0].item() + 3 * 2 * x[1].item():.1f}]")

print("\n" + "=" * 70)
print("示例 8: 批量数据的 grad_outputs")
print("=" * 70)

# 批量输入
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

y = x**2  # shape: (2, 2)

# grad_outputs 也需要匹配形状
grad_outputs = torch.tensor([[1.0, 0.5], [2.0, 1.5]])

y.backward(grad_outputs)

print(f"x shape: {x.shape}")
print(f"y = x^2:\n{y.data}")
print(f"\ngrad_outputs:\n{grad_outputs}")
print(f"\nx.grad:\n{x.grad}")

print("\n逐元素计算:")
print("  x.grad[0,0] = grad_outputs[0,0] * 2*x[0,0] = 1.0 * 2 = 2.0")
print("  x.grad[0,1] = grad_outputs[0,1] * 2*x[0,1] = 0.5 * 4 = 2.0")
print("  x.grad[1,0] = grad_outputs[1,0] * 2*x[1,0] = 2.0 * 6 = 12.0")
print("  x.grad[1,1] = grad_outputs[1,1] * 2*x[1,1] = 1.5 * 8 = 12.0")

print("\n" + "=" * 70)
print("示例 9: 实际应用 - 多任务学习")
print("=" * 70)


class MultiTaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(5, 10)
        self.task1 = nn.Linear(10, 3)
        self.task2 = nn.Linear(10, 2)

    def forward(self, x):
        features = torch.relu(self.shared(x))
        out1 = self.task1(features)
        out2 = self.task2(features)
        return out1, out2


model = MultiTaskNet()
x = torch.randn(4, 5)
y1_true = torch.randn(4, 3)
y2_true = torch.randn(4, 2)

out1, out2 = model(x)

# 分别计算两个任务的损失
loss1 = ((out1 - y1_true) ** 2).sum()
loss2 = ((out2 - y2_true) ** 2).sum()

print(f"任务1 损失: {loss1.item():.4f}")
print(f"任务2 损失: {loss2.item():.4f}")

# 方法1: 加权求和
total_loss = 0.7 * loss1 + 0.3 * loss2
total_loss.backward()

print("\n使用加权和的梯度已计算")

# 方法2: 分别计算梯度然后组合 (等价)
# loss1.backward(torch.tensor(0.7), retain_graph=True)
# loss2.backward(torch.tensor(0.3))

print("\n" + "=" * 70)
print("示例 10: 常见错误和解决方案")
print("=" * 70)

print("错误1: 向量输出不提供 grad_outputs")
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x**2

try:
    y.backward()  # 会报错
except RuntimeError as e:
    print(f"❌ 错误: {str(e)[:60]}...")
    print("   解决: y.backward(torch.ones_like(y))")

print("\n错误2: grad_outputs 形状不匹配")
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x**2

try:
    y.backward(torch.tensor([1.0]))  # 形状不匹配
except RuntimeError:
    print("❌ 错误: grad_outputs 形状必须与输出相同")
    print("   输出形状:", y.shape)
    print("   grad_outputs 应该是:", torch.ones_like(y).shape)

print("\n" + "=" * 70)
print("核心要点总结")
print("=" * 70)
print("""
1. grad_outputs 是链式法则中的"上游梯度"
   dL/dx = dL/dy * dy/dx
           ↑        ↑
      grad_outputs  计算得到

2. 标量输出: grad_outputs 默认为 1.0（不需要指定）

3. 向量输出: 必须指定 grad_outputs（形状要匹配）

4. 实际含义: 
   - 如果 y 是最终损失，grad_outputs = 1
   - 如果 y 是中间结果，grad_outputs = dL/dy（来自后续计算）

5. 应用场景:
   - 计算雅可比矩阵（逐行计算）
   - 向量-雅可比积（VJP）
   - 多任务学习的权重
   - 自定义复杂的梯度流

6. 等价关系:
   y.sum().backward() ≈ y.backward(torch.ones_like(y))
""")
print("=" * 70)
