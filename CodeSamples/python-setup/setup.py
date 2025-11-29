from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# 项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

# C++/CUDA 源文件
cpp_sources = [
    'csrc/my_ops.cpp',
    'csrc/cpu_ops.cpp',
]

cuda_sources = [
    'csrc/cuda/cuda_ops.cu',
    'csrc/cuda/cuda_kernels.cu',
]

# 合并所有源文件
all_sources = cpp_sources + cuda_sources

# 编译参数
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-gencode', 'arch=compute_70,code=sm_70',  # V100
        '-gencode', 'arch=compute_75,code=sm_75',  # T4, RTX 2080
        '-gencode', 'arch=compute_80,code=sm_80',  # A100
        '-gencode', 'arch=compute_86,code=sm_86',  # RTX 3090
    ]
}

# 定义扩展模块
ext_modules = [
    cpp_extension.CUDAExtension(
        name='my_cuda_ops',  # 模块名称
        sources=all_sources,
        include_dirs=[
            os.path.join(project_root, 'csrc'),
            os.path.join(project_root, 'csrc/cuda'),
        ],
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name='my_cuda_extension',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='PyTorch CUDA extension example',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    packages=['my_package'],
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: MIT License',
    ],
)