import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 60)
print("示例 1: 基础梯度计算")
print("=" * 60)

# 创建张量并启用梯度追踪
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

print(f"x = {x.item()}")
print(f"y = x^2 = {y.item()}")

# 反向传播
y.backward()
print(f"dy/dx = 2x = {x.grad.item()}")

print("\n" + "=" * 60)
print("示例 2: 梯度累积问题")
print("=" * 60)

x = torch.tensor([3.0], requires_grad=True)

# 第一次计算
y1 = x ** 2
y1.backward()
print(f"第一次backward后，x.grad = {x.grad.item()}")

# 如果不清零，梯度会累积
y2 = x ** 3
y2.backward()
print(f"第二次backward后（未清零），x.grad = {x.grad.item()}")

# 清零梯度
x.grad.zero_()
y3 = x ** 3
y3.backward()
print(f"清零后再次backward，x.grad = {x.grad.item()}")

print("\n" + "=" * 60)
print("示例 3: 简单线性回归")
print("=" * 60)

# 生成训练数据：y = 3x + 2 + 噪声
torch.manual_seed(42)
x_train = torch.randn(100, 1)
y_label = 3 * x_train + 2 + torch.randn(100, 1) * 0.3

# 定义模型参数
w = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([[0.0]], requires_grad=True)

# 训练参数
learning_rate = 0.01
epochs = 100

# 记录损失
losses = []

print("开始训练...")
for epoch in range(epochs):
    # 前向传播
    y_pred = x_train @ w + b
    
    # 计算损失（均方误差）
    loss = ((y_pred - y_label) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 手动更新参数（梯度下降）
    with torch.no_grad():  # 更新参数时不需要计算梯度
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\n最终参数: w = {w.item():.4f}, b = {b.item():.4f}")
print("真实参数: w = 3.0000, b = 2.0000")

print("\n" + "=" * 60)
print("示例 4: 使用优化器的神经网络")
print("=" * 60)

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练数据：y = sin(x)
x_data = torch.linspace(-3, 3, 100).reshape(-1, 1)
y_data = torch.sin(x_data)

print("使用优化器训练神经网络...")
for epoch in range(200):
    # 前向传播
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    
    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}")

print("\n" + "=" * 60)
print("示例 5: 查看网络中的梯度")
print("=" * 60)

# 创建一个小型网络
net = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

# 前向传播
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[1.0]])
output = net(x)
loss = (output - y) ** 2

# 反向传播
loss.backward()

# 查看每层的梯度
print("网络各层的梯度:")
for name, param in net.named_parameters():
    if param.grad is not None:
        print(f"{name}:")
        print(f"  形状: {param.grad.shape}")
        print(f"  梯度: {param.grad}")
        print()

print("=" * 60)
print("关键要点总结:")
print("=" * 60)
print("1. requires_grad=True: 启用梯度追踪")
print("2. .backward(): 计算梯度")
print("3. .grad: 访问梯度值")
print("4. .zero_grad(): 清零梯度（防止累积）")
print("5. with torch.no_grad(): 临时禁用梯度计算")
print("6. optimizer.step(): 使用计算的梯度更新参数")
print("=" * 60)