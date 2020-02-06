from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='rev_trilinear',
      ext_modules=[CUDAExtension('rev_trilinear', ['rev_trilinear.cpp', 'forward.cu', 'backward.cu'])],
      cmdclass={'build_ext': BuildExtension})
