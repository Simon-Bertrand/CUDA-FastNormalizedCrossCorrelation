
import os
import subprocess
nvcc_path = subprocess.check_output(['where', 'nvcc'], shell=True, text=True).strip().split('\n')[0]
cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
# Définir dans l'environnement avant l'import de torch
os.environ['CUDA_HOME'] = cuda_home
# Forcer la mise à jour pour les sous-processus
import sys
if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.SetEnvironmentVariableW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    kernel32.SetEnvironmentVariableW('CUDA_HOME', cuda_home)
print(f"Auto-detected CUDA_HOME: {cuda_home}")

from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

import glob



ext_modules = []
cuda_kernels_dir = 'cuda_kernels'

if os.path.exists(cuda_kernels_dir):
    for kernel_dir in sorted(os.listdir(cuda_kernels_dir)):
        kernel_path = os.path.join(cuda_kernels_dir, kernel_dir)
        if os.path.isdir(kernel_path):
            # Chercher tous les fichiers .cu dans le dossier et ses sous-dossiers
            cu_files = glob.glob(os.path.join(kernel_path, '**', '*.cu'), recursive=True)
            if not cu_files:
                # Si pas trouvé récursivement, chercher directement dans le dossier
                cu_files = glob.glob(os.path.join(kernel_path, '*.cu'))
            
            if cu_files:
                # Vérifier si cuFFT est nécessaire (chercher #include <cufft.h>)
                libraries = []
                for cu_file in cu_files:
                    try:
                        with open(cu_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if '#include <cufft.h>' in content or '#include "cufft.h"' in content:
                                libraries = ['cufft']
                                break
                    except Exception as e:
                        print(f"Warning: Could not read {cu_file}: {e}")
                
                # Le nom de l'extension est le nom du dossier
                ext_name = kernel_dir
                ext_modules.append(CUDAExtension(ext_name, cu_files, libraries=libraries))
                print(f"Found CUDA extension: {ext_name} with {len(cu_files)} file(s)")

setup(
    name='cuda_kernels',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
