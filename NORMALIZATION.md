# Local Standard Deviation Operator

**Objective**: Implement a highly optimized, differentiable **Local Standard Deviation** operator. 
This operator is the only complex piece required for ZNCC. The rest (Kernel normalization, final division) will be handled by standard PyTorch Autograd.

**Function Signature**: `LocalStandardDeviation(Image, OnesKernel) -> StdMap`

---

## 1. Mathematical Definition

**Forward**:
Given Image $I$ and a Kernel of Ones $\mathbb{1}$ (with size $N$).

1.  **Integrals** (Computed via FFT):
    *   $S_1 = I \star \mathbb{1}$ (Local Sum)
    *   $S_2 = I^2 \star \mathbb{1}$ (Local Sum of Squares)
    
    *Optimization*: Computed in a single Batched `fft_cc` pass with inputs $[I, I^2]$ against shared $\mathbb{1}$.

2.  **Variance**:
    *   $\text{Var} = S_2 - \frac{1}{N} S_1^2$
    *   $\text{StdMap} = \sqrt{\max(0, \text{Var}) + \epsilon}$

**Backward**:
Given gradient $\nabla Std$ ($g$).

1.  **Gradient wrt Variance**:
    *   $\nabla \text{Var} = \frac{g}{2 \cdot \text{StdMap} + \epsilon}$
    
2.  **Gradient wrt Integrals**:
    *   $\nabla S_2 = \nabla \text{Var}$
    *   $\nabla S_1 = \nabla \text{Var} \cdot (-\frac{2}{N} S_1)$

3.  **Gradient wrt Image** (Convolution):
    *   The gradient of a convolution $S = I \star K$ is $\nabla I = \nabla S \star K_{flipped}$. Since $\mathbb{1}$ is symmetric:
    *   $\nabla I_{S1} = \nabla S_1 \star \mathbb{1}$
    *   $\nabla I_{S2} = \nabla S_2 \star \mathbb{1}$
    
    *Optimization*: Computed in a single Batched `fft_cc` pass with inputs $[\nabla S_1, \nabla S_2]$ against shared $\mathbb{1}$.

4.  **Final Combination**:
    *   $\nabla I = \nabla I_{S1} + 2 \cdot I \cdot \nabla I_{S2}$

---

## 2. Implementation Strategy

### A. Python Autograd Function (`LocalStdFunction`)
We will implement a custom `torch.autograd.Function`.

**Forward logic**:
1.  Prepare Batch: `input_stack = cat([I, I**2])`.
2.  Run `fft_cc(input_stack, ones_kernel)`.
3.  Compute `StdMap` (using fused CUDA kernel or JIT if possible).
4.  Save `I`, `S1`, `StdMap` for backward.

**Backward logic**:
1.  Compute `dVar`, `dS1`, `dS2`.
2.  Prepare Batch: `grad_stack = cat([dS1, dS2])`.
3.  Run `fft_cc(grad_stack, ones_kernel)`.
4.  Combine to get `dI`.

### B. High-Performance Kernels (C++/CUDA)
To ensure "highly optimized" execution, we will implement two lightweight CUDA kernels to handle the element-wise complexities and fusion.

1.  **`local_std_forward_kernel`**:
    *   **Input**: $S_1, S_2$
    *   **Output**: $StdMap$
    *   **Logic**: `sqrt(max(0, s2 - s1*s1/N) + eps)`
    *   *Why*: Reduces memory read/write overhead compared to 4 separate PyTorch ops.

2.  **`local_std_backward_kernel`**:
    *   **Input**: $g_{std}, StdMap, S_1, I, \nabla I_{S1}, \nabla I_{S2}$
    *   **Output**: $\nabla I$
    *   **Logic**: 
        *   Intermediate computation of $\nabla S1, \nabla S2$ is implicit? No, we need $\nabla S1, \nabla S2$ explicitly to run the FFT.
        *   So we actually need **two** backward kernels:
        
        *   **Kernel B1 (Prepare Grads)**:
            *   Input: $g_{std}, StdMap, S_1$
            *   Output: $\nabla S_1, \nabla S_2$
            
        *   **Kernel B2 (Combine Grads)**:
            *   Input: $\nabla I_{S1}, \nabla I_{S2}, I$
            *   Output: $\nabla I$
            *   Logic: `dIso1 + 2*I*dIso2`


