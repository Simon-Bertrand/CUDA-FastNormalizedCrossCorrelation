# Gradients Derivation for Autograd

This document details the mathematical derivation of the gradients (Backward Pass) for the Cross-Correlation (CC) and Zero-Normalized Cross-Correlation (ZNCC) operators implemented in this library.

## 1. Notations

- Let $I$ be the input image of size $H \times W$.
- Let $T$ be the template (kernel) of size $h \times w$.
- Let $N = h \times w$ be the number of elements in the template.
- Let $O$ be the output tensor (result of the correlation).
- The correlation operation is defined with "valid" padding (no padding), so the output size is $(H-h+1) \times (W-w+1)$.

We denote a window of the image centered (or anchored) at position $u$ as $I_u$. In the implementation, the anchor is top-left, so $I_u$ refers to the patch of size $h \times w$ starting at $u$.
Index $v$ iterates over the spatial domain of the kernel ($0 \le v_y < h, 0 \le v_x < w$).

## 2. Cross-Correlation (CC)

### Forward Pass
The standard Cross-Correlation is defined as:
$$
O(u) = (I \star T)(u) = \sum_{v} I(u+v) \cdot T(v)
$$

### Backward Pass

Let $L$ be the scalar loss function. We are given the upstream gradient $\frac{\partial L}{\partial O}$.
We need to compute the gradients with respect to the inputs: $\frac{\partial L}{\partial I}$ and $\frac{\partial L}{\partial T}$.

#### Gradient w.r.t Image ($I$)
By the chain rule:
$$
\frac{\partial L}{\partial I(k)} = \sum_{u} \frac{\partial L}{\partial O(u)} \frac{\partial O(u)}{\partial I(k)}
$$
From the definition of $O(u)$, $\frac{\partial O(u)}{\partial I(k)} = T(k-u)$ if $k-u$ is a valid index in $T$, and 0 otherwise.
Let $v = k-u$, so $u = k-v$.
$$
\frac{\partial L}{\partial I(k)} = \sum_{v} \frac{\partial L}{\partial O(k-v)} \cdot T(v)
$$
This operation corresponds to the **convolution** of the upstream gradient $\frac{\partial L}{\partial O}$ with the kernel $T$ (flipped if strictly speaking convolution, but usually implemented as correlation with appropriate indexing).
$$
\frac{\partial L}{\partial I} = \frac{\partial L}{\partial O} \ast T
$$
(Where $\ast$ denotes full convolution).

#### Gradient w.r.t Template ($T$)
$$
\frac{\partial L}{\partial T(v)} = \sum_{u} \frac{\partial L}{\partial O(u)} \frac{\partial O(u)}{\partial T(v)}
$$
From $O(u)$, $\frac{\partial O(u)}{\partial T(v)} = I(u+v)$.
$$
\frac{\partial L}{\partial T(v)} = \sum_{u} \frac{\partial L}{\partial O(u)} \cdot I(u+v)
$$
This operation corresponds to the correlation between the Image $I$ and the upstream gradient $\frac{\partial L}{\partial O}$.
$$
\frac{\partial L}{\partial T} = I \star \frac{\partial L}{\partial O}
$$

---

## 3. Zero-Normalized Cross-Correlation (ZNCC)

This is the more complex operator.

### 3.1 Definitions & Forward Pass

**Step 1: Template Standardization**
The template $T$ is globally standardized.
Let $\mu_{T}$ be the mean of $T$ and $\sigma_{T}$ be the standard deviation (uncorrected).
$$
\mu_{T} = \frac{1}{N} \sum_v T(v)
$$
$$
\sigma_{T} = \sqrt{ \frac{1}{N} \sum_v (T(v) - \mu_{T})^2 }
$$
The standardized template is:
$$
\hat{T}(v) = \frac{T(v) - \mu_{T}}{\sigma_{T}}
$$
Note that $\sum_v \hat{T}(v) = 0$ and $\sum_v \hat{T}(v)^2 = N$.

**Step 2: Local Image Standardization**
For each window $u$, we calculate the local mean $\mu_{I_u}$ and local standard deviation $\sigma_{I_u}$.
$$
\mu_{I_u} = \frac{1}{N} \sum_v I(u+v)
$$
$$
\sigma_{I_u} = \sqrt{ \frac{1}{N} \sum_v (I(u+v) - \mu_{I_u})^2 }
$$
The locally standardized image patch is:
$$
\hat{I}_{u}(v) = \frac{I(u+v) - \mu_{I_u}}{\sigma_{I_u}}
$$

**Step 3: ZNCC Calculation**
The ZNCC score at $u$ is the dot product of the standardized vectors divided by $N$.
The implementation effectively computes:
$$
Z(u) = \frac{1}{\sigma_{T} \sigma_{I_u}} \sum_v (I(u+v) - \mu_{I_u})(T(v) - \mu_{T})
$$
Using the fact that $\sum (T(v)-\mu_{T}) = 0$, the term involving $\mu_{I_u}$ vanishes:
$$
\sum_v (I(u+v) - \mu_{I_u})(T(v) - \mu_{T}) = \sum_v I(u+v)(T(v) - \mu_{T}) - \mu_{I_u} \cdot 0
$$
So:
$$
Z(u) = \frac{1}{\sigma_{I_u}} \sum_v I(u+v) \hat{T}(v)
$$
(Note: The factor $1/\sigma_{T}$ is absorbed into $\hat{T}$).

Let $C(u) = \sum_v I(u+v) \hat{T}(v)$. This is the standard cross-correlation of $I$ with the pre-standardized template $\hat{T}$.
Then:
$$
Z(u) = \frac{C(u)}{\sigma_{I_u}}
$$

### 3.2 Backward Pass Derivation

We need $\nabla_{I} L$ and $\nabla_{T} L$.
Let $\delta(u) = \frac{\partial L}{\partial Z(u)}$.

#### Gradient w.r.t Image ($I$)

We analyze the contribution of a single output pixel $Z(u)$ to the gradient of the input pixel $I(k)$.
$$
\frac{\partial Z(u)}{\partial I(k)} = \frac{\partial}{\partial I(k)} \left( C(u) \sigma_{I_u}^{-1} \right)
$$
$$
= \sigma_{I_u}^{-1} \frac{\partial C(u)}{\partial I(k)} + C(u) \frac{\partial \sigma_{I_u}^{-1}}{\partial I(k)}
$$

**Term A:** $\frac{\partial C(u)}{\partial I(k)}$
$$
C(u) = \sum_v \hat{T}(v) I(u+v)
$$
Derivative is $\hat{T}(k-u)$ (if $k$ is in window $u$).

**Term B:** $\frac{\partial \sigma_{I_u}^{-1}}{\partial I(k)}$
$$
\frac{\partial \sigma_{I_u}^{-1}}{\partial I(k)} = -\frac{1}{\sigma_{I_u}^2} \frac{\partial \sigma_{I_u}}{\partial I(k)} = -\frac{1}{2\sigma_{I_u}^3} \frac{\partial \sigma_{I_u}^2}{\partial I(k)}
$$
Recall variance: $\sigma_{I_u}^2 = \frac{1}{N} \sum_v I(u+v)^2 - (\mu_{I_u})^2$.
$$
\frac{\partial \sigma_{I_u}^2}{\partial I(k)} = \frac{1}{N} \cdot 2 I(k) - 2 \mu_{I_u} \frac{\partial \mu_{I_u}}{\partial I(k)}
$$
Since $\mu_{I_u} = \frac{1}{N} \sum I$, $\frac{\partial \mu_{I_u}}{\partial I(k)} = \frac{1}{N}$.
$$
\frac{\partial \sigma_{I_u}^2}{\partial I(k)} = \frac{2}{N} (I(k) - \mu_{I_u})
$$
Substituting back:
$$
\frac{\partial \sigma_{I_u}^{-1}}{\partial I(k)} = -\frac{1}{2\sigma_{I_u}^3} \frac{2}{N} (I(k) - \mu_{I_u}) = -\frac{1}{N \sigma_{I_u}^2} \hat{I}_{u}(k-u)
$$

**Combining Terms:**
$$
\frac{\partial Z(u)}{\partial I(k)} = \frac{1}{\sigma_{I_u}} \hat{T}(k-u) + C(u) \left( -\frac{1}{N \sigma_{I_u}^2} \hat{I}_{u}(k-u) \right)
$$
Since $Z(u) = C(u) / \sigma_{I_u}$:
$$
\frac{\partial Z(u)}{\partial I(k)} = \frac{1}{\sigma_{I_u}} \left( \hat{T}(k-u) - \frac{Z(u)}{N} \hat{I}_{u}(k-u) \right)
$$

**Total Gradient:**
$$
\frac{\partial L}{\partial I(k)} = \sum_u \delta(u) \frac{\partial Z(u)}{\partial I(k)}
$$
$$
\frac{\partial L}{\partial I(k)} = \sum_u \frac{\delta(u)}{\sigma_{I_u}} \left( \hat{T}(k-u) - \frac{Z(u)}{N} \hat{I}_{u}(k-u) \right)
$$

This can be interpreted as:
1. Backpropagate $\frac{\delta(u)}{\sigma_{I_u}}$ through the correlation with $\hat{T}$.
2. Subtract the backpropagation of $\frac{\delta(u) \cdot Z(u)}{N \sigma_{I_u}}$ through the "local standardization" operation (conceptually similar to instance norm backward).

#### Gradient w.r.t Template ($T$)

$Z(u)$ depends on $T$ via $\hat{T}(v)$.
$$
Z(u) = \sum_v \hat{I}_{u}(v) \hat{T}(v)
$$

Let $\nabla_{\hat{T}} L$ be the gradient w.r.t the standardized template.
$$
\frac{\partial L}{\partial \hat{T}(v)} = \sum_u \delta(u) \frac{\partial Z(u)}{\partial \hat{T}(v)} = \sum_u \delta(u) \hat{I}_{u}(v)
$$
This is the correlation of the input image $I$ (locally standardized) with the upstream gradient $\delta$.

Now we backpropagate from $\hat{T}$ to $T$.
$$
\hat{T}(v) = \frac{T(v) - \mu_{T}}{\sigma_{T}}
$$
The Jacobian of Standardization is standard for Batch/Layer Norm:
$$
\frac{\partial L}{\partial T(j)} = \frac{1}{\sigma_{T}} \left( \frac{\partial L}{\partial \hat{T}(j)} - \frac{1}{N} \sum_v \frac{\partial L}{\partial \hat{T}(v)} - \frac{\hat{T}(j)}{N} \sum_v \frac{\partial L}{\partial \hat{T}(v)} \hat{T}(v) \right)
$$

Let $G_{\hat{T}} = \nabla_{\hat{T}} L$.
$$
\nabla_{T} L = \frac{1}{\sigma_{T}} \left( G_{\hat{T}} - \mathrm{mean}(G_{\hat{T}}) - \hat{T} \cdot \langle G_{\hat{T}}, \hat{T} \rangle_{\mathrm{avg}} \right)
$$

### Summary of ZNCC Backward Steps

1. Compute $G_{\hat{T}} = \sum_u \delta(u) \hat{I}_{u}$.
2. Compute $\nabla_{T} L$ by backpropagating $G_{\hat{T}}$ through template standardization.
3. Compute update map $M(u) = \frac{\delta(u)}{\sigma_{I_u}}$.
4. Compute $\nabla_{I} L$ by convolving $M$ with $\hat{T}$ and subtracting the "local norm" correction term.
