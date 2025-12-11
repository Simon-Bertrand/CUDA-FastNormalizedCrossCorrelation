To complete the implementation:

- Implement the CPU version
- Implement CC autograd backward GPU and CPU (Using conv and conv transpose)
- Implement ZNCC autograd backward GPU and CPU (hardest)

They are available (as slower vesion) in Python with : https://github.com/Simon-Bertrand/FastCrossCorr-PyTorch


In this implementation: we don't use integral images to compute the denominator of the ZNCC to avoid possible numerical precision issue that
can happens with the double cumsum operator on float32.

Instead, we stack all data needed for the ZNCC (I, I², T, 1.) and do the whole Fourier hadamard product using a single cuFFT call.
The ZNCC can be wrote :

```math
ZNCC(\mathbf{x})
=
\frac{
    % Numerator: Correlation of Image with Centered Template
    % Note: The term (I - \mu_I) simplifies to I here because \sum (T-\mu_T) = 0.
    (I \star (T-\mu_T))(\mathbf{x})
}{
    \sqrt{
        \left[
            % Image Centered Energy (Variance * N)
            (I^2 \star \mathbf{1})(\mathbf{x})
            -
            \frac{(I \star \mathbf{1})(\mathbf{x})^2}{N}
        \right]
        \cdot
        % Template Centered Energy (must be squared norm)
        
    } ||T-\mu_T||_2
}
```
In practice, we can directly compute if we standardize the template  T (i.e., enforce zero mean and unit L2 norm): 
```math
\hat{T} = \frac{T - \mu_T}{||T - \mu_T||_2}
```
Then

```math
ZNCC'(\mathbf{x}) = \frac{(I \star \hat{T})(\mathbf{x})}
{
    \sqrt{
        (I^2 \star \mathbf{1})(\mathbf{x})
        -
        \frac{(I \star \mathbf{1})(\mathbf{x})^2}{N}
    }
}
```
This rewritten equation highlights that, after standardizing the template, the denominator only normalizes with respect to the local statistics of the image I and the template only enters linearly in the numerator as a normalized filter.
