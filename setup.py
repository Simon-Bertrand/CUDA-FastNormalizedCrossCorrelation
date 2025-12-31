
import os
import subprocess
import glob
import sys
import shutil
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

# Detect CUDA
nvcc_path = shutil.which('nvcc')
cuda_home = None
if nvcc_path:
    cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
    os.environ['CUDA_HOME'] = cuda_home
    print(f"Auto-detected CUDA_HOME: {cuda_home}")
else:
    print("CUDA (nvcc) not found. Building in CPU-only mode.")

ext_modules = []
cuda_kernels_dir = 'cuda_kernels'

if os.path.exists(cuda_kernels_dir):
    for kernel_dir in sorted(os.listdir(cuda_kernels_dir)):
        kernel_path = os.path.join(cuda_kernels_dir, kernel_dir)
        if os.path.isdir(kernel_path):
            # Sources
            sources = []

            # Helper to find files
            def find_files(pattern):
                files = glob.glob(os.path.join(kernel_path, '**', pattern), recursive=True)
                if not files:
                     files = glob.glob(os.path.join(kernel_path, pattern))
                return files

            cu_files = find_files('*.cu')
            cpp_files = find_files('*.cpp')

            # Determine if we can build with CUDA
            has_cuda_code = len(cu_files) > 0
            build_with_cuda = has_cuda_code and (cuda_home is not None)

            # Construct sources list
            sources.extend(cpp_files)
            if build_with_cuda:
                sources.extend(cu_files)

            if not sources:
                continue

            libraries = []
            define_macros = []
            extra_compile_args = {}

            if build_with_cuda:
                define_macros.append(('WITH_CUDA', None))
                libraries.append('cufft')

                if os.name == 'nt':
                    extra_compile_args = {'cxx': ['/std:c++17'], 'nvcc': ['-std=c++17']}
                else:
                    extra_compile_args = {'cxx': ['-std=c++17'], 'nvcc': ['-std=c++17']}

                # Check for cufft in sources if needed (legacy check)
                # But we know we use it.
            else:
                 # If we have CUDA files but no CUDA compiler, we skip them.
                 # Warn user
                 if has_cuda_code:
                     print(f"Warning: CUDA files found in {kernel_dir} but nvcc is missing. Compiling CPU only.")
            
            # Create Extension
            ext_name = kernel_dir
            if build_with_cuda:
                ext_modules.append(CUDAExtension(
                    ext_name,
                    sources,
                    libraries=libraries,
                    define_macros=define_macros,
                    extra_compile_args=extra_compile_args
                ))
            else:
                ext_modules.append(CppExtension(
                    ext_name,
                    sources,
                    define_macros=define_macros,
                    extra_compile_args=extra_compile_args
                ))
                
            print(f"Added extension: {ext_name} (CUDA={'On' if build_with_cuda else 'Off'})")

setup(
    name='cuda_kernels',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
