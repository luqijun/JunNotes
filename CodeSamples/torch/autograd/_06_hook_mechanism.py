import torch
import torch.nn as nn

print("=" * 70)
print("示例 1: Tensor Hook - 监控梯度")
print("=" * 70)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)


# 定义 hook 函数
def grad_hook(grad):
    print(f"  梯度被计算: {grad}")
    # 可以修改梯度
    return grad * 2  # 梯度翻倍


# 注册 hook
hook_handle = x.register_hook(grad_hook)

z = (x * y).sum()
print(f"z = (x * y).sum() = {z.item()}")

print("\n反向传播:")
z.backward()

print(f"\nx 的最终梯度: {x.grad}")
print(f"y 的最终梯度: {y.grad}")
print("注意: x 的梯度被翻倍了!")

# 移除 hook
hook_handle.remove()

print("\n" + "=" * 70)
print("示例 2: Module Hook - 前向传播")
print("=" * 70)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleNet()

# 存储中间激活值
activations = {}


def forward_hook(module, input, output):
    """前向传播 hook"""
    # module: 当前层
    # input: 输入 (tuple)
    # output: 输出
    name = module.__class__.__name__
    activations[name] = output.detach()
    print(f"  {name}: input shape={input[0].shape}, output shape={output.shape}")


# 为每一层注册 hook
for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        layer.register_forward_hook(forward_hook)

x = torch.randn(2, 3)
print("前向传播:")
output = model(x)

print("\n保存的激活值:")
for name, act in activations.items():
    print(f"  {name}: shape={act.shape}, mean={act.mean().item():.4f}")

print("\n" + "=" * 70)
print("示例 3: Module Hook - 反向传播")
print("=" * 70)

gradients = {}


def backward_hook(module, grad_input, grad_output):
    """反向传播 hook"""
    # grad_input: 相对于输入的梯度 (tuple)
    # grad_output: 相对于输出的梯度 (tuple)
    name = module.__class__.__name__
    gradients[name] = {
        'grad_output': grad_output[0].detach() if grad_output[0] is not None else None,
        'grad_input': grad_input[0].detach() if grad_input[0] is not None else None,
    }
    print(f"  {name}: grad_output shape={grad_output[0].shape if grad_output[0] is not None else None}")


model = SimpleNet()

# 注册反向 hook
for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        layer.register_full_backward_hook(backward_hook)

x = torch.randn(2, 3)
output = model(x)
loss = output.sum()

print("反向传播:")
loss.backward()

print("\n保存的梯度信息:")
for name, grads in gradients.items():
    print(f"  {name}:")
    if grads['grad_output'] is not None:
        print(f"    grad_output shape: {grads['grad_output'].shape}")
    if grads['grad_input'] is not None:
        print(f"    grad_input shape: {grads['grad_input'].shape}")

print("\n" + "=" * 70)
print("示例 4: 梯度裁剪 Hook")
print("=" * 70)


def gradient_clipping_hook(grad, max_norm=1.0):
    """梯度裁剪 hook"""
    norm = grad.norm()
    if norm > max_norm:
        print(f"  梯度范数 {norm:.4f} 超过 {max_norm}, 进行裁剪")
        return grad * (max_norm / norm)
    return grad


x = torch.tensor([10.0], requires_grad=True)
x.register_hook(lambda grad: gradient_clipping_hook(grad, max_norm=1.0))

y = x**3
print("y = x^3, x=10")
print(f"y = {y.item()}")

print("\n反向传播 (带梯度裁剪):")
y.backward()
print(f"裁剪后的梯度: {x.grad.item():.4f}")
print(f"原始梯度应该是: 3*x^2 = {3 * 10**2}")

print("\n" + "=" * 70)
print("示例 5: 梯度监控和可视化")
print("=" * 70)


class GradientMonitor:
    """梯度监控器"""

    def __init__(self):
        self.gradients = {}
        self.hooks = []

    def register(self, model):
        """为模型所有参数注册 hook"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(lambda grad, name=name: self.save_grad(name, grad))
                self.hooks.append(hook)

    def save_grad(self, name, grad):
        """保存梯度统计"""
        self.gradients[name] = {
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'min': grad.min().item(),
            'max': grad.max().item(),
            'norm': grad.norm().item(),
        }

    def report(self):
        """报告梯度统计"""
        print("\n梯度统计报告:")
        print("-" * 70)
        print(f"{'层名':<20} {'均值':<10} {'标准差':<10} {'范数':<10}")
        print("-" * 70)
        for name, stats in self.gradients.items():
            print(f"{name:<20} {stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['norm']:<10.4f}")

    def clear(self):
        """清除保存的梯度"""
        self.gradients.clear()

    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()


# 创建模型和监控器
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 1))

monitor = GradientMonitor()
monitor.register(model)

# 训练一步
x = torch.randn(5, 10)
y = torch.randn(5, 1)
output = model(x)
loss = ((output - y) ** 2).mean()

print("前向传播完成，开始反向传播...")
loss.backward()

monitor.report()

print("\n" + "=" * 70)
print("示例 6: Pre-Hook (输入修改)")
print("=" * 70)


def pre_hook(module, input):
    """前向传播前的 hook，可以修改输入"""
    print(f"  原始输入范围: [{input[0].min():.2f}, {input[0].max():.2f}]")
    # 归一化输入
    normalized = (input[0] - input[0].mean()) / (input[0].std() + 1e-8)
    print(f"  归一化后范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
    return (normalized,)


model = nn.Linear(5, 3)
model.register_forward_pre_hook(pre_hook)

x = torch.randn(2, 5) * 100  # 大范围的输入
print("前向传播 (带输入归一化):")
output = model(x)

print("\n" + "=" * 70)
print("示例 7: 梯度累积检测")
print("=" * 70)


class GradientAccumulationDetector:
    """检测意外的梯度累积"""

    def __init__(self):
        self.grad_counts = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self.check_accumulation(name, grad))

    def check_accumulation(self, name, grad):
        self.grad_counts[name] = self.grad_counts.get(name, 0) + 1
        if self.grad_counts[name] > 1:
            print(f"  ⚠️  警告: {name} 的梯度被计算了 {self.grad_counts[name]} 次!")

    def reset(self):
        self.grad_counts.clear()


model = nn.Linear(3, 2)
detector = GradientAccumulationDetector()
detector.register(model)

x = torch.randn(1, 3)

print("第一次反向传播:")
loss1 = model(x).sum()
loss1.backward()

print("\n第二次反向传播 (未清零梯度):")
loss2 = model(x).sum()
loss2.backward()

print("\n检测到梯度累积问题!")
print("解决方案: 使用 optimizer.zero_grad() 或 param.grad.zero_()")

print("\n" + "=" * 70)
print("示例 8: 条件梯度修改")
print("=" * 70)


def conditional_grad_modification(grad, threshold=1.0):
    """根据条件修改梯度"""
    # 对异常大的梯度进行标记和修改
    large_grad_mask = grad.abs() > threshold
    if large_grad_mask.any():
        print(f"  检测到 {large_grad_mask.sum().item()} 个异常梯度")
        # 将大梯度裁剪到阈值
        grad = torch.where(large_grad_mask, grad.sign() * threshold, grad)
    return grad


x = torch.tensor([1.0, 2.0, 10.0], requires_grad=True)
x.register_hook(lambda grad: conditional_grad_modification(grad, threshold=5.0))

y = (x**3).sum()
print("反向传播 (条件梯度修改):")
y.backward()
print(f"修改后的梯度: {x.grad}")
print("原始梯度应该是: [3, 12, 300]")

print("\n" + "=" * 70)
print("Hook 使用最佳实践")
print("=" * 70)
print("1. 记得移除不再需要的 hook (调用 handle.remove())")
print("2. Hook 函数应该简洁高效，避免复杂计算")
print("3. 注意 hook 的执行顺序 (按注册顺序)")
print("4. 修改梯度时要小心，可能影响训练稳定性")
print("5. 使用 hook 进行调试，但生产环境中移除")
print("6. Forward hook 不能修改输出，只能观察")
print("7. Backward hook 可以修改梯度，需谨慎使用")
print("=" * 70)
