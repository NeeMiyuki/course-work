import torch
import numpy as np
from numba import cuda
from numba import njit

@cuda.jit
def rgk4_kneading_kernel(output, aStart, aEnd, aCount, bStart, bEnd, bCount, c, dt, N, stride, kneadingStart, kneadingEnd):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    total_Param_Space_Size = aCount * bCount

    if idx < total_Param_Space_Size:
        aIdx = idx % aCount
        bIdx = idx // aCount

        a = aStart + (aEnd - aStart) * aIdx / (aCount - 1)
        b = bStart + (bEnd - bStart) * aIdx / (bCount - 1)

        kernel_result = a + b + c + dt + stride + kneadingStart + kneadingEnd
        output[idx] = kernel_result

@njit
def ode_function(t, y):
    return -y

@njit
def rgk4(function_to_solve, t, y, dt):
    dt2 = dt/2.0
    dt6 = dt/6.0

    k1 = function_to_solve(t,y)
    k2 = function_to_solve(t + dt2, y + dt2 * k1)
    k3 = function_to_solve(t + dt2, y + dt2 * k2)
    k4 = function_to_solve(t + dt, y + dt * k3)

    y_return = y + dt6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return y_return

@njit
def solve(function_to_solve, y0, t0, tn, dt):
    steps = int((tn - t0) / dt) + 1
    result = np.zeros((steps, y0.size), dtype=np.float64)
    time = np.zeros(steps, dtype=np.float64)

    y = y0.copy()
    t = t0

    for i in range(steps):
        result[i, :] = y
        time[i] = t

        y = rgk4(function_to_solve, t, y ,dt)
        t += dt

    return time, result

aStart, aEnd, aCount = 0.1, 2.0, 100
bStart, bEnd, bCount = 0.1, 2.0, 100
c = 5.7
dt = 0.01
N = 10000
stride = 100

kneadingStart = 0
kneadingEnd = 10

total_Param_space_size = aCount * bCount

output = np.zeros(total_Param_space_size, dtype = np.float64)
outputGPU = cuda.to_device(output)

threads_per_block = 512
blocks = 3
grid_size = blocks *  threads_per_block

rgk4_kneading_kernel[blocks, threads_per_block](outputGPU, aStart, aEnd, aCount, bStart, bEnd, bCount, c, dt, N, stride, kneadingStart, kneadingEnd)
output = outputGPU

y0 = torch.tensor([1.0], dtype = torch.float64, device="cuda")
t0 = 0.0
tn = 1.0
dt = 0.1

for i in range(total_Param_space_size):
    y0_cpu = np.array([output[i]], dtype = np.float64)
    y0_gpu = torch.tensor(y0_cpu, device="cuda")

    time, result = solve(ode_function, y0_cpu, t0, tn, dt)

    time_cpu = torch.tensor(time).cpu().numpy()
    result_cpu = torch.tensor(result).cpu().numpy()

    print(f"Parameter set {i+1}: a + b = {output[i]:.4f}")
    for t, y in zip(time_cpu, result_cpu):
        print(f"t = {t:.2f}, y = {y[0]:.4f}")
    print("\n")