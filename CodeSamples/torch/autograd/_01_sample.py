import torch

# 创建一个需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)

# 定义一个简单函数：y = x^2
y = x ** 2

print(f"x = {x}")
print(f"y = {y}")

# 反向传播计算梯度
y.backward()

# 查看梯度：dy/dx = 2x = 2*2 = 4
print(f"x的梯度: {x.grad}")