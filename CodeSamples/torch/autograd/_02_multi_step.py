import torch

x = torch.tensor([3.0], requires_grad=True)

# 多步运算：z = (x + 2) * (x^2)
y = x + 2          # y = x + 2 = 5
z = y * (x ** 2)   # z = 5 * 9 = 45

print(f"z = {z}")

z.backward()

# 梯度计算：dz/dx = d[(x+2)*x²]/dx = x² + 2x(x+2) = x² + 2x² + 4x = 3x² + 4x
# 当 x=3: 3*9 + 4*3 = 27 + 12 = 39
print(f"x的梯度: {x.grad}")