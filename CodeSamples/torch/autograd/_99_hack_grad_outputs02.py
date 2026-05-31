import torch

print("=" * 70)
print("可视化理解 grad_outputs: 计算图视角")
print("=" * 70)

print("""
场景: 完整的神经网络计算图

输入(x) -> 中间层(h) -> 输出层(y) -> 损失(L)

backward() 从右向左传播梯度:

L.backward()
  ↓
计算 dL/dy (=1, 因为 L 对自己的导数是 1)
  ↓
y.backward(dL/dy)  ← 这里的参数就是 grad_outputs
  ↓
计算 dL/dh = dL/dy * dy/dh
  ↓
h.backward(dL/dh)  ← dL/dh 作为 grad_outputs 继续传播
  ↓
计算 dL/dx = dL/dh * dh/dx
""")

print("\n" + "=" * 70)
print("具体示例: 完整前向和反向过程")
print("=" * 70)

# 构建一个简单的计算图
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"1. 输入 x = {x.data}")

# 第一层: h = x^2
h = x**2
print(f"2. 中间层 h = x^2 = {h.data}")

# 第二层: y = h + 1
y = h + 1
print(f"3. 输出层 y = h + 1 = {y.data}")

# 损失: L = sum(y)
L = y.sum()
print(f"4. 损失 L = sum(y) = {L.item()}")

print("\n" + "-" * 70)
print("反向传播过程:")
print("-" * 70)

# 手动模拟反向传播
print("\n步骤1: 从 L 开始")
print("  dL/dL = 1.0")

print("\n步骤2: L 对 y 的梯度")
dL_dy = torch.ones_like(y)
print(f"  dL/dy = {dL_dy.data}")
print("  (因为 L = sum(y), 所以每个 y[i] 的贡献都是 1)")

print("\n步骤3: 计算 y 对 h 的梯度")
print("  dy/dh = 1 (因为 y = h + 1)")

print("\n步骤4: 链式法则 - L 对 h 的梯度")
dL_dh = dL_dy * 1  # dy/dh = 1
print(f"  dL/dh = dL/dy * dy/dh = {dL_dh.data}")

print("\n步骤5: 计算 h 对 x 的梯度")
dh_dx = 2 * x.data
print(f"  dh/dx = 2*x = {dh_dx}")

print("\n步骤6: 链式法则 - L 对 x 的梯度")
dL_dx_manual = dL_dh * dh_dx
print(f"  dL/dx = dL/dh * dh/dx = {dL_dx_manual}")

print("\n" + "-" * 70)
print("使用 PyTorch 自动计算:")
print("-" * 70)
L.backward()
print(f"  x.grad = {x.grad}")
print("  ✓ 与手动计算结果一致!")

print("\n" + "=" * 70)
print("模拟 backward() 的内部实现")
print("=" * 70)


def manual_backward_demo():
    """手动模拟 backward 过程，展示 grad_outputs 的作用"""

    x = torch.tensor([[1.0, 2.0]], requires_grad=True)

    # 前向传播
    h = x**2
    y = h + 1
    L = y.sum()

    print("如果我们调用 y.backward()...")
    print("PyTorch 会报错，因为 y 不是标量")
    print("\n但我们可以调用 y.backward(grad_outputs)")
    print("其中 grad_outputs 就是 dL/dy")

    # 手动指定 grad_outputs
    dL_dy = torch.ones_like(y)

    print(f"\ngrad_outputs = dL/dy = {dL_dy.data}")

    x.grad = None  # 清零
    y.backward(dL_dy)

    print(f"结果: x.grad = {x.grad}")

    # 与直接对 L backward 比较
    x.grad = None
    h = x**2
    y = h + 1
    L = y.sum()
    L.backward()

    print(f"\n直接 L.backward(): x.grad = {x.grad}")
    print("两种方式结果相同!")


manual_backward_demo()

print("\n" + "=" * 70)
print("实际例子: 分段计算梯度")
print("=" * 70)

# 场景: 计算图太大，需要分段计算
x = torch.tensor([2.0], requires_grad=True)

# 第一段计算
print("第一段: y = x^2")
y = x**2
print(f"  y = {y.item()}")

# 假设后续有复杂计算，我们知道 dL/dy = 3.0
dL_dy = torch.tensor([3.0])
print(f"\n从后续计算得到: dL/dy = {dL_dy.item()}")

# 使用 grad_outputs 完成反向传播
y.backward(dL_dy)
print(f"反向传播: x.grad = {x.grad.item()}")

print("\n验证:")
print("  dL/dx = dL/dy * dy/dx")
print(f"        = {dL_dy.item()} * 2*x")
print(f"        = {dL_dy.item()} * {2 * x.item()}")
print(f"        = {dL_dy.item() * 2 * x.item()}")

print("\n" + "=" * 70)
print("实际应用: 注意力机制中的 grad_outputs")
print("=" * 70)


def attention_example():
    """注意力机制中的梯度传播"""

    # Q, K, V 矩阵
    Q = torch.randn(2, 4, requires_grad=True)
    K = torch.randn(2, 4, requires_grad=True)
    V = torch.randn(2, 4, requires_grad=True)

    # 计算注意力分数
    scores = Q @ K.t()  # (2, 2)
    print(f"注意力分数形状: {scores.shape}")

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    print(f"注意力权重形状: {attn_weights.shape}")

    # 加权求和
    output = attn_weights @ V  # (2, 4)
    print(f"输出形状: {output.shape}")

    # 假设从后续层传来的梯度
    grad_output = torch.ones_like(output)
    print(f"\ngrad_outputs 形状: {grad_output.shape}")

    # 反向传播
    output.backward(grad_output)

    print("\n各参数的梯度:")
    print(f"  Q.grad 形状: {Q.grad.shape}")
    print(f"  K.grad 形状: {K.grad.shape}")
    print(f"  V.grad 形状: {V.grad.shape}")


attention_example()

print("\n" + "=" * 70)
print("常见场景对照表")
print("=" * 70)

scenarios = [
    ("训练最后一层", "loss.backward()", "标量损失", "不需要"),
    ("中间层激活", "h.backward(grad)", "向量", "需要(来自后层)"),
    ("多输出网络", "y.backward(grad)", "多维张量", "需要(任务权重)"),
    ("雅可比计算", "y.backward(e_i)", "向量", "需要(单位向量)"),
    ("梯度检查", "y.backward(v)", "向量", "需要(随机向量)"),
]

print(f"{'场景':<15} {'调用方式':<20} {'输出类型':<12} {'grad_outputs'}")
print("-" * 70)
for scenario in scenarios:
    print(f"{scenario[0]:<15} {scenario[1]:<20} {scenario[2]:<12} {scenario[3]}")

print("\n" + "=" * 70)
print("记忆口诀")
print("=" * 70)
print("""
1. grad_outputs 是"上游的梯度"
2. 标量输出，grad_outputs 默认是 1
3. 向量输出，grad_outputs 必须给定
4. 形状必须匹配输出的形状
5. 本质是链式法则的实现

公式记忆:
  当前梯度 = grad_outputs × 局部梯度
  dL/dx = (dL/dy) × (dy/dx)
          ↑         ↑
      grad_outputs  自动计算
""")

print("\n" + "=" * 70)
print("调试技巧")
print("=" * 70)

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x**2

# 技巧1: 检查输出形状
print(f"1. 输出形状: {y.shape}")
print(f"   -> grad_outputs 应该也是: {y.shape}")

# 技巧2: 使用全1作为默认
grad_outputs = torch.ones_like(y)
print("\n2. 默认使用: grad_outputs = ones_like(y)")

# 技巧3: 理解含义
print("\n3. 含义: 我们关心每个输出元素对损失的等同贡献")

y.backward(grad_outputs)
print(f"\n4. 结果: x.grad = {x.grad}")

print("=" * 70)
