from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
# os.environ['CC']= '/usr/local/gcc-5.3.0/bin/g++'
# os.environ['CXX']= '/usr/local/gcc-5.3.0/bin/g++'
# CC = os.environ.get("CC", None)
# CC_arg = "-ccbin={}".format(CC)

setup(
    name='crop_split_cuda',
    ext_modules=[
        CUDAExtension('crop_split_cuda', [
            'src/crop_split_cuda.cpp',
            'src/crop_split_cuda_kernel.cu',
        ],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
                # CC_arg,
                
            ]
        })
    ],
    cmdclass={'build_ext': BuildExtension})

setup(
    name='crop_split_gt_cuda',
    ext_modules=[
        CUDAExtension('crop_split_gt_cuda', [
            'src/crop_split_gt_cuda.cpp',
            'src/crop_split_gt_cuda_kernel.cu',
        ],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
                # CC_arg,
                
            ]
        })
    ],
    cmdclass={'build_ext': BuildExtension})
