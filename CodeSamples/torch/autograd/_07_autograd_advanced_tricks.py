import time

import torch
import torch.nn as nn
from torch.autograd import Function

# ======================================================================
# 技巧 1: 混合精度训练中的自定义函数
# ======================================================================
# 输入 (FP16): tensor([ 0.1000,  1.0000, 10.0000], dtype=torch.float16, requires_grad=True)
# 输出 (FP16): tensor([-2.3027,  0.0000,  2.3027], dtype=torch.float16,
#        grad_fn=<MixedPrecisionOperationBackward>)
# 梯度 (FP16): tensor([10.0000,  1.0000,  0.1000], dtype=torch.float16)
print("=" * 70)
print("技巧 1: 混合精度训练中的自定义函数")
print("=" * 70)


class MixedPrecisionOperation(Function):
    """在 FP16 训练中保持某些操作的 FP32 精度"""

    @staticmethod
    def forward(ctx, x):
        # 转换到 FP32 进行高精度计算
        x_fp32 = x.float()
        result = torch.log(x_fp32 + 1e-8)  # 数值稳定的 log
        ctx.save_for_backward(x)
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # 在 FP32 中计算梯度
        x_fp32 = x.float()
        grad_input = grad_output.float() / (x_fp32 + 1e-8)
        return grad_input.to(x.dtype)


x = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float16, requires_grad=True)
y = MixedPrecisionOperation.apply(x)

print(f"输入 (FP16): {x}")
print(f"输出 (FP16): {y}")

y.sum().backward()
print(f"梯度 (FP16): {x.grad}")

# ======================================================================
# 技巧 2: 梯度检查点 (Gradient Checkpointing)
# ======================================================================
# 不使用检查点:
#   时间: 0.1083s

# 使用检查点:
#   时间: 0.3041s
#   (可能更慢，但显著节省内存)
print("\n" + "=" * 70)
print("技巧 2: 梯度检查点 (Gradient Checkpointing)")
print("=" * 70)


def checkpoint_function(func, *args):
    """
    简化版的梯度检查点
    前向传播时不保存中间激活，反向传播时重新计算
    """

    class CheckpointFunction(Function):
        @staticmethod
        def forward(ctx, *inputs):
            # 不保存中间结果
            ctx.func = func
            with torch.no_grad():
                outputs = func(*inputs)
            # 只保存输入用于重新计算
            ctx.save_for_backward(*inputs)
            return outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            inputs = ctx.saved_tensors
            # 重新计算前向传播（这次保存计算图）
            with torch.enable_grad():
                detached_inputs = [x.detach().requires_grad_(True) for x in inputs]
                outputs = ctx.func(*detached_inputs)

            # 计算梯度
            torch.autograd.backward(outputs, grad_outputs)
            return tuple(x.grad for x in detached_inputs)

    return CheckpointFunction.apply(*args)


# 示例: 深层网络
def heavy_computation(x):
    """模拟计算密集型操作"""
    for _ in range(100):
        x = torch.relu(x)
        x = x + 0.01
    return x


x = torch.randn(1000, 1000, requires_grad=True)

print("不使用检查点:")
start = time.time()
y = heavy_computation(x)
loss = y.sum()
loss.backward()
time_normal = time.time() - start
print(f"  时间: {time_normal:.4f}s")

x.grad = None
print("\n使用检查点:")
start = time.time()
y = checkpoint_function(heavy_computation, x)
loss = y.sum()
loss.backward()
time_checkpoint = time.time() - start
print(f"  时间: {time_checkpoint:.4f}s")
print("  (可能更慢，但显著节省内存)")

# ======================================================================
# 技巧 3: 分离部分计算图
# ======================================================================
# 训练判别器 - 生成器梯度应该被阻断:
#   D 损失: 2.4889
#   生成器有梯度? False (应该是 False)
# 训练生成器 - 判别器梯度应该被阻断:
#   G 损失: -2.4889
#   生成器有梯度? True (应该是 True)
#   判别器有梯度? True (应该是 False)
print("\n" + "=" * 70)
print("技巧 3: 分离部分计算图")
print("=" * 70)


# 场景: 生成对抗网络 (GAN) 训练
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


G = Generator()
D = Discriminator()

z = torch.randn(5, 10)

print("训练判别器 - 生成器梯度应该被阻断:")
fake_data = G(z)
# 分离生成器输出，避免梯度流回生成器
fake_data_detached = fake_data.detach()
d_loss = D(fake_data_detached).sum()
print(f"  D 损失: {d_loss.item():.4f}")
d_loss.backward()

# 检查生成器参数是否有梯度
g_has_grad = any(p.grad is not None for p in G.parameters())
print(f"  生成器有梯度? {g_has_grad} (应该是 False)")

print("\n训练生成器 - 判别器梯度应该被阻断:")
for p in D.parameters():
    p.requires_grad = False  # 冻结判别器

fake_data = G(z)
g_loss = -D(fake_data).sum()  # 生成器想要欺骗判别器
print(f"  G 损失: {g_loss.item():.4f}")

for p in G.parameters():
    p.grad = None
g_loss.backward()

g_has_grad = any(p.grad is not None for p in G.parameters())
d_has_grad = any(p.grad is not None for p in D.parameters())
print(f"  生成器有梯度? {g_has_grad} (应该是 True)")
print(f"  判别器有梯度? {d_has_grad} (应该是 False)")

# ======================================================================
# 技巧 4: 梯度累积实现大批次训练
# ======================================================================
# 模拟批次大小: 32
# 实际小批次: 8
# 累积步数: 4
#   步骤 1: loss=1.5185
#   步骤 2: loss=1.1364
#   步骤 3: loss=1.6141
#   步骤 4: loss=1.4942
# 参数更新完成
print("\n" + "=" * 70)
print("技巧 4: 梯度累积实现大批次训练")
print("=" * 70)

model = nn.Linear(5, 3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 模拟大批次 (batch_size=32) 但内存只够处理 batch_size=8
real_batch_size = 32
mini_batch_size = 8
accumulation_steps = real_batch_size // mini_batch_size

print(f"模拟批次大小: {real_batch_size}")
print(f"实际小批次: {mini_batch_size}")
print(f"累积步数: {accumulation_steps}")

# 生成数据
data = torch.randn(real_batch_size, 5)
targets = torch.randn(real_batch_size, 3)

optimizer.zero_grad()

for i in range(accumulation_steps):
    # 取小批次
    start_idx = i * mini_batch_size
    end_idx = start_idx + mini_batch_size

    mini_batch = data[start_idx:end_idx]
    mini_targets = targets[start_idx:end_idx]

    # 前向传播
    output = model(mini_batch)
    loss = ((output - mini_targets) ** 2).mean()

    # 累积梯度 (除以累积步数以得到平均梯度)
    (loss / accumulation_steps).backward()

    print(f"  步骤 {i + 1}: loss={loss.item():.4f}")

# 一次性更新参数
optimizer.step()
print("参数更新完成")

# ======================================================================
# 技巧 5: 多任务学习的梯度平衡
# ======================================================================
# 任务1损失: 2.0551
# 任务2损失: 0.7745
# 任务1梯度范数: 2.5237
# 任务2梯度范数: 1.3568
# 动态权重: w1=0.3496, w2=0.6782
print("\n" + "=" * 70)
print("技巧 5: 多任务学习的梯度平衡")
print("=" * 70)


class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(10, 20)
        self.task1_head = nn.Linear(20, 5)
        self.task2_head = nn.Linear(20, 3)

    def forward(self, x):
        shared_features = torch.relu(self.shared(x))
        out1 = self.task1_head(shared_features)
        out2 = self.task2_head(shared_features)
        return out1, out2


model = MultiTaskModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(4, 10)
y1 = torch.randn(4, 5)
y2 = torch.randn(4, 3)

# 计算两个任务的损失
out1, out2 = model(x)
loss1 = ((out1 - y1) ** 2).mean()
loss2 = ((out2 - y2) ** 2).mean()

print(f"任务1损失: {loss1.item():.4f}")
print(f"任务2损失: {loss2.item():.4f}")

# 方法1: 简单加权
# loss_total = 0.5 * loss1 + 0.5 * loss2

# 方法2: 梯度归一化
optimizer.zero_grad()

# 分别计算梯度
loss1.backward(retain_graph=True)
grad1_norm = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).norm()

optimizer.zero_grad()
loss2.backward(retain_graph=True)
grad2_norm = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).norm()

print(f"\n任务1梯度范数: {grad1_norm.item():.4f}")
print(f"任务2梯度范数: {grad2_norm.item():.4f}")

# 根据梯度范数动态调整权重
w1 = 1.0 / (grad1_norm.item() + 1e-8)
w2 = 1.0 / (grad2_norm.item() + 1e-8)
w1 = w1 / (w1 + w2)
w2 = w2 / (w1 + w2)

print(f"动态权重: w1={w1:.4f}, w2={w2:.4f}")

optimizer.zero_grad()
loss_total = w1 * loss1 + w2 * loss2
loss_total.backward()
optimizer.step()

# ======================================================================
# 技巧 6: 自定义梯度缩放
# ======================================================================
# 正常反向传播:
#   梯度: tensor([2., 4.])
# 梯度反转:
#   梯度: tensor([-2., -4.]) (注意负号)
print("\n" + "=" * 70)
print("技巧 6: 自定义梯度缩放")
print("=" * 70)


class GradientRescale(Function):
    """自定义梯度缩放，常用于对抗训练"""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


# 应用: 梯度反转层 (用于域适应)
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


x = torch.tensor([1.0, 2.0], requires_grad=True)

# 正常梯度
y1 = (x**2).sum()

# 梯度反转
grad_reverse = GradientReversalLayer.apply
y2 = grad_reverse((x**2).sum(), 1.0)

print("正常反向传播:")
y1.backward()
print(f"  梯度: {x.grad}")

x.grad = None
print("\n梯度反转:")
y2.backward()
print(f"  梯度: {x.grad} (注意负号)")

# ======================================================================
# 技巧 7: 条件计算图
# ======================================================================
# 条件为 True:
#   输出: 1.9872
#   梯度: 2.5217
# 条件为 False:
#   输出: 1.0000
#   梯度: 2.0000
print("\n" + "=" * 70)
print("技巧 7: 条件计算图")
print("=" * 70)


def conditional_forward(x, condition):
    """根据条件选择不同的计算路径"""
    if condition:
        # 路径 A: 复杂计算
        return torch.sigmoid(x) * torch.exp(x)
    else:
        # 路径 B: 简单计算
        return x**2


x = torch.tensor([1.0], requires_grad=True)

print("条件为 True:")
y1 = conditional_forward(x, True)
y1.backward()
print(f"  输出: {y1.item():.4f}")
print(f"  梯度: {x.grad.item():.4f}")

x.grad = None
print("\n条件为 False:")
y2 = conditional_forward(x, False)
y2.backward()
print(f"  输出: {y2.item():.4f}")
print(f"  梯度: {x.grad.item():.4f}")

# ======================================================================
# 技巧 8: 参数组的分别优化
# ======================================================================
# 不同层的学习率:
#   组 0: lr=0.001
#   组 1: lr=0.0001
print("\n" + "=" * 70)
print("技巧 8: 参数组的分别优化")
print("=" * 70)

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

# 为不同层设置不同学习率
optimizer = torch.optim.Adam(
    [{'params': model[0].parameters(), 'lr': 1e-3}, {'params': model[2].parameters(), 'lr': 1e-4}]
)

x = torch.randn(5, 10)
y = model(x)
loss = y.sum()

print("不同层的学习率:")
for i, param_group in enumerate(optimizer.param_groups):
    print(f"  组 {i}: lr={param_group['lr']}")

loss.backward()
optimizer.step()

print("\n" + "=" * 70)
print("高级技巧总结")
print("=" * 70)
print("1. 混合精度: 关键操作保持 FP32 精度")
print("2. 梯度检查点: 用计算换内存")
print("3. 分离计算图: 控制梯度流向")
print("4. 梯度累积: 模拟大批次训练")
print("5. 多任务平衡: 动态调整任务权重")
print("6. 梯度缩放: 自定义梯度大小")
print("7. 条件计算: 动态选择计算路径")
print("8. 参数组优化: 不同参数不同策略")
print("=" * 70)


# ======================================================================
# 高级技巧总结
# ======================================================================
# 1. 混合精度: 关键操作保持 FP32 精度
# 2. 梯度检查点: 用计算换内存
# 3. 分离计算图: 控制梯度流向
# 4. 梯度累积: 模拟大批次训练
# 5. 多任务平衡: 动态调整任务权重
# 6. 梯度缩放: 自定义梯度大小
# 7. 条件计算: 动态选择计算路径
# 8. 参数组优化: 不同参数不同策略
# ======================================================================
