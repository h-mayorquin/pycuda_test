import numpy as np
import pycuda.autoinit 
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__  void error(float *y, float *y_est, float *error)
{
const int i = threadIdx.x;
error[i] = (y_est[i] - y[i]);
}
""")


def linear_function(w, x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = np.array((1, x))
            return np.sum(np.dot(w, x))
        else:
            return np.dot(w, x)

    else:
        x = np.array((1, x))
        return np.dot(w, x)

# W's
w0t = 1
w1t = 2
w = np.array((w0t, w1t))

# X formation
x1 = np.arange(-10, 11, 0.1)
Npoints = 200
x1 = np.random.normal(size=Npoints)
x0 = np.ones_like(x1)
x = np.vstack((x0, x1)).astype(np.float32)
y = linear_function(w, x).astype(np.float32)

w_est = np.array((1, 2))

y_est_host = linear_function(w_est, x)
error_host = np.zeros_like(y)
print('Error before opreation', error_host)

# Device arrays
y_est_device = cuda.mem_alloc(y_est_host.nbytes)
error_device = cuda.mem_alloc(error_host.nbytes)
y_device = cuda.mem_alloc(y.nbytes)

# Memory allocations
cuda.memcpy_htod(y_device, y)
cuda.memcpy_htod(y_est_device, y_est_host)
cuda.memcpy_htod(error_device, error_host)

# Perform the function
function_error = mod.get_function("error")
function_error(y_device, y_est_device, error_device, block=(1024, 1, 1))

# Copy back to host
cuda.memcpy_dtoh(error_host, error_device)
print('Error after operation', error_host)
