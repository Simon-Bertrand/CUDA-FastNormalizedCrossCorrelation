
import torch
import pytest
import time
from torch_bidimcc import local_std

def run_benchmark():
    device = "cpu"
    dtype = torch.float64
    B, C = 2, 2
    H, W = 256, 256
    h, w = 32, 32

    print(f"Allocating tensors: {B}x{C}x{H}x{W} dtype={dtype}")
    image = torch.randn(B, C, H, W, dtype=dtype, device=device, requires_grad=True)
    ones = torch.ones(B, C, h, w, dtype=dtype, device=device, requires_grad=False)

    grad_output = torch.randn(B, C, H - h + 1, W - w + 1, dtype=dtype, device=device)

    print("Warmup...")
    for _ in range(3):
        out = local_std(image, ones, 1e-5)
        out.backward(grad_output, retain_graph=True)
        image.grad.zero_()

    print("Benchmark...")
    start_time = time.perf_counter()
    iterations = 10
    for i in range(iterations):
        print(f"Iteration {i}")
        out = local_std(image, ones, 1e-5)
        out.backward(grad_output, retain_graph=True)

    duration = (time.perf_counter() - start_time) / iterations
    print(f"Duration: {duration}")

if __name__ == "__main__":
    run_benchmark()
