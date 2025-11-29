# PyTorch C++/CUDA Extension 示例项目

这是一个完整的示例项目，展示如何使用 `setup.py` 编译 C++ 和 CUDA 文件，并将它们集成到 PyTorch 中。

## 项目结构

```
project/
├── setup.py                   # 编译配置文件
├── README.md                  # 项目文档
├── test_ops.py                # 测试脚本
├── my_package/                # Python 包
│   └── __init__.py            # 包初始化和包装函数
└── csrc/                      # C++/CUDA 源代码
    ├── my_ops.cpp             # Python 绑定
    ├── cpu_ops.cpp            # CPU 实现
    └── cuda/
        ├── cuda_ops.cu        # CUDA 接口
        └── cuda_kernels.cu    # CUDA kernels
```

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA Toolkit (如果需要 GPU 支持)
- GCC/G++ 编译器
- NVCC (NVIDIA CUDA 编译器)

**注意**：

- 确保你的系统已安装 CUDA Toolkit，且版本与 PyTorch 兼容。例如，PyTorch 1.13.1 要求 CUDA Toolkit 11.7, PyTorch 2.3.0 要求 CUDA Toolkit 11.8。
- 若之前安装torch没有在对应的CUDA版本环境下进行安装，则可能会导致编译错误，建议先卸载之前的torch，然后在对应的CUDA版本环境下重新安装torch。

## 安装步骤

### 1. 克隆或创建项目

```bash
mkdir my_cuda_project
cd my_cuda_project
# 将所有文件放置在相应目录中
```

### 2. 安装依赖

```bash
pip install torch numpy
```

### 3. 构建/编译扩展

#### 3.1 构建扩展
```bash
python setup.py build_ext --inplace
```

#### 3.2 编译选项

有几种编译方式：

##### 方式 1: 开发模式安装（推荐）
```bash
python setup.py develop
```

##### 方式 2: 正式安装
```bash
python setup.py install
```

##### 方式 3: 使用 pip 安装
```bash
pip install -e .
```

### 4. 运行测试

```bash
python test_ops.py
```

## 使用示例

### 基本用法

```python
import torch
import my_package

# CPU 操作
a = torch.randn(100, 100)
b = torch.randn(100, 100)
c = my_package.add(a, b)

# GPU 操作
if torch.cuda.is_available():
    a_gpu = torch.randn(100, 100, device='cuda')
    b_gpu = torch.randn(100, 100, device='cuda')
    c_gpu = my_package.add(a_gpu, b_gpu)
```

### 自定义操作

```python
import torch
import my_package

x = torch.randn(100, 100)
output1, output2 = my_package.custom_op(x, scale=2.0)
```

### 支持自动微分

```python
import torch
import my_package

a = torch.randn(100, 100, requires_grad=True)
b = torch.randn(100, 100, requires_grad=True)

c = my_package.custom_add_with_grad(a, b)
loss = c.sum()
loss.backward()

print(a.grad)  # 梯度已计算
```

## 关键特性

1. **CPU/GPU 自动切换**: 根据输入张量自动选择 CPU 或 CUDA 实现
2. **类型模板化**: 支持 float 和 double 类型
3. **自动微分支持**: 集成 PyTorch 的 autograd 系统
4. **性能优化**: 使用 CUDA 加速计算密集型操作

## 编译选项说明

在 `setup.py` 中可以配置：

- **计算能力 (Compute Capability)**: 根据你的 GPU 型号调整
  - sm_70: V100
  - sm_75: T4, RTX 2080
  - sm_80: A100
  - sm_86: RTX 3090

- **编译优化**:
  - `-O3`: 最高级别优化
  - `--use_fast_math`: 使用快速数学库
  - `-std=c++17`: 使用 C++17 标准

## 常见问题

### 1. 编译错误：找不到 CUDA

确保安装了 CUDA Toolkit 并设置了环境变量：
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2. 导入错误

如果出现 `ImportError`，尝试重新编译：
```bash
python setup.py clean --all
python setup.py develop
```

### 3. CUDA 版本不匹配

确保 PyTorch 和 CUDA Toolkit 版本兼容：
```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### 4. 找不到THC/THC.h文件

在PyTorch 1.11及以上版本中，开发者常遇到fatal error: THC/THC.h: No such file or directory编译错误。该问题源于PyTorch架构升级。参考[PyTorch新版本THC头文件缺失问题解决方案](https://comate.baidu.com/zh/page/6725yrbq5tm)

## 扩展开发

要添加新的操作：

1. 在 `csrc/cuda/cuda_kernels.cu` 中实现 CUDA kernel
2. 在 `csrc/cuda/cuda_ops.cu` 中添加包装函数
3. 在 `csrc/cpu_ops.cpp` 中添加 CPU 版本
4. 在 `csrc/my_ops.cpp` 中添加 Python 绑定
5. 在 `my_package/__init__.py` 中暴露接口

## 许可证

MIT License

## 参考资源

- [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Custom Operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)