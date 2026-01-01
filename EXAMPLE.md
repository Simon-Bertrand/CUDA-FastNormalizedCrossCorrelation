```python
!uv pip install . --no-build-isolation --force-reinstall
```

    [2mUsing Python 3.12.3 environment at:...
    


```python
import torch
import torch_cireg

H,W = 256,256
h,w = 64,64

a = 5* torch.rand(3,4,H, H, device='cuda')
b = 5* torch.rand(3,4,h, w, device='cuda')

```


```python
%timeit torch_cireg.fft_cc(a,b)
```

    162 μs ± 1.58 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    


```python
%timeit torch_cireg.fft_zncc(a,b)
```

    591 μs ± 7.44 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    


```python

import matplotlib.pyplot as plt
a = 5*torch.rand(4,3,256, 256, device='cuda')
b =5* torch.rand(4,3,64, 64, device='cuda')

# Compute both results
f_cuda = torch_cireg.fft_cc(a, b)
f_cuda_2d = f_cuda[0, 0].detach().cpu().numpy()  # first channel/batch

f_naive = (a.unfold(-2, 64, 1).unfold(-2, 64, 1) * b.unsqueeze(-3).unsqueeze(-3)).sum((-2, -1))
f_naive_2d = f_naive[0, 0].detach().cpu().numpy()  # first channel/batch
# Pad f_naive_2d to match the size of f_cuda_2d
import numpy as np


diff_2d = f_cuda_2d - f_naive_2d

fig, axs = plt.subplots(1, 3, figsize=(21, 6))

im0 = axs[0].imshow(f_cuda_2d, aspect='auto')
axs[0].set_title("torch_cireg.fft_cc(a, b)")
axs[0].set_xlabel("Window x")
axs[0].set_ylabel("Window y")
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

im1 = axs[1].imshow(f_naive_2d, aspect='auto')
axs[1].set_title("Naive Unfold/Sum")
axs[1].set_xlabel("Window x")
axs[1].set_ylabel("Window y")
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

im2 = axs[2].imshow(diff_2d, aspect='auto', cmap='bwr')
axs[2].set_title("Diff (cuda - naive)")
axs[2].set_xlabel("Window x")
axs[2].set_ylabel("Window y")
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()




```


    
![png](public/dbg_4_0.png)
    

