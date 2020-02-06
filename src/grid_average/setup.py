from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='grid_average',
      ext_modules=[CUDAExtension('grid_average', ['grid_average.cpp', 'forward.cu', 'forward_feature.cu'])],
      cmdclass={'build_ext': BuildExtension})
