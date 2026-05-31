import torch
import torch.autograd as autograd

# ======================================================================
# 示例 1: 计算高阶导数
# ======================================================================
# f(x) = x^4, x=2
# f(2) = 16.0
# f'(2) = 4*2^3 = 32.0
# f''(2) = 12*2^2 = 48.0
# f'''(2) = 24*2 = 48.0
# f''''(2) = 24 = 24.0
print("=" * 70)
print("示例 1: 计算高阶导数")
print("=" * 70)

# 函数: f(x) = x^4
x = torch.tensor([2.0], requires_grad=True)

# 零阶: f(x)
y = x ** 4
print("f(x) = x^4, x=2")
print(f"f(2) = {y.item()}")

# 一阶导数: f'(x) = 4x^3
dy_dx = autograd.grad(y, x, create_graph=True)[0]
print(f"f'(2) = 4*2^3 = {dy_dx.item()}")

# 二阶导数: f''(x) = 12x^2
d2y_dx2 = autograd.grad(dy_dx, x, create_graph=True)[0]
print(f"f''(2) = 12*2^2 = {d2y_dx2.item()}")

# 三阶导数: f'''(x) = 24x
d3y_dx3 = autograd.grad(d2y_dx2, x, create_graph=True)[0]
print(f"f'''(2) = 24*2 = {d3y_dx3.item()}")

# 四阶导数: f''''(x) = 24
d4y_dx4 = autograd.grad(d3y_dx3, x)[0]
print(f"f''''(2) = 24 = {d4y_dx4.item()}")

# ======================================================================
# 示例 2: 计算 Hessian 矩阵
# ======================================================================
# 函数: f(x,y) = x² + 2xy + 3y²
# 在点 x=1.0, y=2.0
# f = 17.0
# Hessian 矩阵:
# tensor([[2., 2.],
#         [2., 6.]])
# 理论 Hessian:
#   ∂²f/∂x² = 2
#   ∂²f/∂x∂y = 2
#   ∂²f/∂y∂x = 2
#   ∂²f/∂y² = 6
print("\n" + "=" * 70)
print("示例 2: 计算 Hessian 矩阵")
print("=" * 70)

def compute_hessian(f, x):
    """
    计算标量函数 f 关于向量 x 的 Hessian 矩阵
    Hessian[i,j] = ∂²f/∂xi∂xj
    """
    n = x.shape[0]
    hessian = torch.zeros(n, n)
    
    # 计算一阶导数
    grad = autograd.grad(f, x, create_graph=True)[0]
    
    # 对每个一阶导数分量再求导
    for i in range(n):
        grad2 = autograd.grad(grad[i], x, retain_graph=True)[0]
        hessian[i] = grad2
    
    return hessian

# 函数: f(x, y) = x^2 + 2xy + 3y^2
x = torch.tensor([1.0, 2.0], requires_grad=True)
f = x[0]**2 + 2*x[0]*x[1] + 3*x[1]**2

print("函数: f(x,y) = x² + 2xy + 3y²")
print(f"在点 x={x[0].item()}, y={x[1].item()}")
print(f"f = {f.item()}")

hessian = compute_hessian(f, x)
print("\nHessian 矩阵:")
print(hessian)

print("\n理论 Hessian:")
print("  ∂²f/∂x² = 2")
print("  ∂²f/∂x∂y = 2")
print("  ∂²f/∂y∂x = 2")
print("  ∂²f/∂y² = 6")


# ======================================================================
# 示例 3: 使用 functorch 高效计算 Hessian
# ======================================================================
# Rosenbrock 函数在 x=tensor([0.5000, 0.5000])
# f = 6.5000
# 梯度: tensor([-51.,  50.])
# Hessian 矩阵:
# tensor([[ 102., -200.],
#         [-200.,  200.]])
print("\n" + "=" * 70)
print("示例 3: 使用 functorch 高效计算 Hessian")
print("=" * 70)

def rosenbrock(x):
    """Rosenbrock 函数: f(x,y) = (1-x)² + 100(y-x²)²"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

x = torch.tensor([0.5, 0.5], requires_grad=True)
f_val = rosenbrock(x)

print(f"Rosenbrock 函数在 x={x.data}")
print(f"f = {f_val.item():.4f}")

# 计算梯度
grad = autograd.grad(f_val, x, create_graph=True)[0]
print(f"\n梯度: {grad.data}")

# 计算 Hessian
hessian = compute_hessian(f_val, x)
print("\nHessian 矩阵:")
print(hessian)

# ======================================================================
# 示例 4: Jacobian 向量积 (JVP)
# ======================================================================
# 输入 x = tensor([2., 3.])
# 方向 v = tensor([1., 1.])
# 输出 f(x) = tensor([7., 6.])
# JVP (J@v) = tensor([7., 3.])
# 手动计算验证:
# J = [[2*x[0], 1    ],
#      [x[1],   x[0] ]]
# J = [[4.0, 1.0],
#      [3.0, 2.0]]
# v^T @ J = [7.0, 3.0]
print("\n" + "=" * 70)
print("示例 4: Jacobian 向量积 (JVP)")
print("=" * 70)

def f(x):
    """向量函数 R² -> R²"""
    return torch.stack([
        x[0]**2 + x[1],
        x[0] * x[1]
    ])

x = torch.tensor([2.0, 3.0], requires_grad=True)
v = torch.tensor([1.0, 1.0])  # 方向向量

# 计算 JVP: J(x) @ v
y = f(x)
jvp = autograd.grad(y, x, grad_outputs=v, create_graph=True)[0]

print(f"输入 x = {x.data}")
print(f"方向 v = {v}")
print(f"输出 f(x) = {y.data}")
print(f"JVP (J@v) = {jvp.data}")

print("\n手动计算验证:")
print("J = [[2*x[0], 1    ],")
print("     [x[1],   x[0] ]]")
print(f"J = [[{2*x[0].item():.1f}, {1.0:.1f}],")
print(f"     [{x[1].item():.1f}, {x[0].item():.1f}]]")
print(f"v^T @ J = [{v[0]*2*x[0].item() + v[1]*x[1].item():.1f}, {v[0]*1.0 + v[1]*x[0].item():.1f}]")

# ======================================================================
# 示例 5: 向量 Jacobian 积 (VJP)
# ======================================================================
# 输入 x = tensor([2., 3.])
# 方向 v = tensor([1., 1.])
# VJP (v^T@J) = tensor([7., 3.])
# 手动计算验证:
# v^T@J = [1, 1] @ J = [7.0, 3.0]
print("\n" + "=" * 70)
print("示例 5: 向量 Jacobian 积 (VJP)")
print("=" * 70)

x = torch.tensor([2.0, 3.0], requires_grad=True)
v = torch.tensor([1.0, 1.0])

y = f(x)

# 计算 VJP: v^T @ J(x)
vjp = autograd.grad(y, x, grad_outputs=v)[0]

print(f"输入 x = {x.data}")
print(f"方向 v = {v}")
print(f"VJP (v^T@J) = {vjp.data}")

print("\n手动计算验证:")
print(f"v^T@J = [1, 1] @ J = [{2*x[0].item() + x[1].item():.1f}, {1 + x[0].item():.1f}]")

# ======================================================================
# 示例 6: 拉普拉斯算子 (Laplacian)
# ======================================================================
# 函数: f(x,y) = x² + y²
# 拉普拉斯: ∇²f = ∂²f/∂x² + ∂²f/∂y² = 4.0
# 理论值: 2 + 2 = 4
print("\n" + "=" * 70)
print("示例 6: 拉普拉斯算子 (Laplacian)")
print("=" * 70)

def compute_laplacian(f, x):
    """
    计算标量函数的拉普拉斯: ∇²f = Σ ∂²f/∂xi²
    """
    grad = autograd.grad(f, x, create_graph=True)[0]
    
    laplacian = 0
    for i in range(x.shape[0]):
        grad2 = autograd.grad(grad[i], x, retain_graph=True)[0]
        laplacian += grad2[i]
    
    return laplacian

# 函数: f(x,y) = x² + y²
x = torch.tensor([1.0, 2.0], requires_grad=True)
f = (x ** 2).sum()

laplacian = compute_laplacian(f, x)
print("函数: f(x,y) = x² + y²")
print(f"拉普拉斯: ∇²f = ∂²f/∂x² + ∂²f/∂y² = {laplacian.item()}")
print("理论值: 2 + 2 = 4")

# ======================================================================
# 示例 7: 牛顿法优化 (使用 Hessian)
# ======================================================================
# 使用牛顿法优化 Rosenbrock 函数
# 目标: 找到最小值点 (1, 1)
# 迭代 0: x=tensor([1., 0.]), f(x)=100.000000
# 迭代 2: x=tensor([1., 1.]), f(x)=0.000000
# 迭代 4: x=tensor([1., 1.]), f(x)=0.000000
# 最优解: tensor([1., 1.])
# 函数值: 0.000000
print("\n" + "=" * 70)
print("示例 7: 牛顿法优化 (使用 Hessian)")
print("=" * 70)

def newton_method(f, x0, n_iterations=10):
    """使用牛顿法最小化函数"""
    x = x0.clone().requires_grad_(True)
    
    history = [x0.clone().detach()]
    
    for i in range(n_iterations):
        # 计算函数值
        f_val = f(x)
        
        # 计算梯度
        grad = autograd.grad(f_val, x, create_graph=True)[0]
        
        # 计算 Hessian
        hessian = compute_hessian(f_val, x)
        
        # 牛顿更新: x = x - H^(-1) @ g
        with torch.no_grad():
            delta = torch.linalg.solve(hessian, grad)
            x = x - delta
            x.requires_grad_(True)
        
        history.append(x.clone().detach())
        
        if i % 2 == 0:
            print(f"迭代 {i}: x={x.data}, f(x)={f(x).item():.6f}")
    
    return x.detach(), history

# 优化 Rosenbrock 函数
x0 = torch.tensor([0.0, 0.0], requires_grad=True)
print("使用牛顿法优化 Rosenbrock 函数")
print("目标: 找到最小值点 (1, 1)")

x_opt, history = newton_method(rosenbrock, x0, n_iterations=6)
print(f"\n最优解: {x_opt}")
print(f"函数值: {rosenbrock(x_opt).item():.6f}")
print("理论最优: [1, 1], f=0")

print("\n" + "=" * 70)
print("关键概念总结")
print("=" * 70)
print("1. create_graph=True: 创建高阶导数的计算图")
print("2. retain_graph=True: 保留图以便多次 backward")
print("3. Hessian: 二阶导数矩阵，用于优化和分析")
print("4. JVP: Jacobian-Vector Product, 高效的前向模式")
print("5. VJP: Vector-Jacobian Product, 高效的反向模式")
print("6. 拉普拉斯: 二阶导数的迹，物理中常用")
print("7. 牛顿法: 使用 Hessian 的二阶优化方法")
print("=" * 70)