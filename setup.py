from setuptools import setup, find_packages
from setuptools.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from Cython.Build import cythonize
import numpy as np
import os.path as osp

ROOT = osp.dirname(osp.abspath(__file__))

# --- Marching Cubes Extension ---
mcubes_ext = Extension(
    "marching_cubes._mcubes",
    [
        "thirdparty/marching_cubes/src/_mcubes.pyx",
        "thirdparty/marching_cubes/src/pywrapper.cpp",
        "thirdparty/marching_cubes/src/marching_cubes.cpp"
    ],
    language="c++",
    extra_compile_args=['-std=c++11', '-Wall'],
    include_dirs=[np.get_include()],
    depends=[
        "thirdparty/marching_cubes/src/marching_cubes.h",
        "thirdparty/marching_cubes/src/pyarray_symbol.h",
        "thirdparty/marching_cubes/src/pyarraymodule.h",
        "thirdparty/marching_cubes/src/pywrapper.h"
    ],
)

# --- Droid Backends CUDA Extension ---
droid_ext = CUDAExtension(
    'droid_backends',
    include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
    sources=[
        'src/lib/droid.cpp',
        'src/lib/droid_kernels.cu',
        'src/lib/correlation_kernels.cu',
        'src/lib/altcorr_kernel.cu',
    ],
    extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': [
            '-O3',
            '-gencode=arch=compute_60,code=sm_60',
            '-gencode=arch=compute_61,code=sm_61',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_89,code=sm_89',
        ]
    }
)

# --- Lietorch CUDA Extension ---
lietorch_ext = CUDAExtension(
    'lietorch_backends',
    include_dirs=[
        osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'),
        osp.join(ROOT, 'thirdparty/eigen')],
    sources=[
        'thirdparty/lietorch/lietorch/src/lietorch.cpp',
        'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
        'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp'],
    extra_compile_args={
        'cxx': ['-O2'],
        'nvcc': [
            '-O2',
            '-gencode=arch=compute_60,code=sm_60',
            '-gencode=arch=compute_61,code=sm_61',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_89,code=sm_89',
        ]
    }
)

setup(
    name='mneslam_extensions',
    version='1.0',
    description='Extensions for MNESLAM system',
    packages=find_packages(where='thirdparty'),
    package_dir={'': 'thirdparty'},
    ext_modules=cythonize([mcubes_ext]) + [droid_ext, lietorch_ext],
    cmdclass={'build_ext': BuildExtension}
)
