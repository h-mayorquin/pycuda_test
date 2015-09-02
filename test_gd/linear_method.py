import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate the data
visualize = False
verbose = False


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


def calculate_error(w, x, y):
    est_y = linear_function(w, x)
    diff = est_y - y
    error = np.sum(diff ** 2)

    return np.average(error)

# W's
w0t = 1
w1t = 2
w = np.array((w0t, w1t))

# X formation
x1 = np.arange(-100, 101, 0.1)
Npoints = 200
x1 = np.random.normal(size=Npoints)
x0 = np.ones_like(x1)
x = np.vstack((x0, x1))

# Noise
sigma = 1.0
noise = sigma * np.random.randn(x1.size)

y = linear_function(w, x)
ynoise = y + noise

error = calculate_error(w, x, y)
error_sigma = calculate_error(w, x, ynoise)

if visualize:
    plt.plot(x1, y)
    plt.hold(True)
    plt.show()

##############
# Now we do the algorithms here
##############

random_int = np.random.randint(low=0, high=Npoints, size=1)
Niterations = 1000
gamma = 1.0

w0est = 0.5
w1est = 0.5
w_est = np.array((w0est, w1est))

for i in range(Niterations):
    # Get the values
    random_int = np.random.randint(low=0, high=Npoints, size=1)
    xi = x[:, random_int]
    yi = y[random_int]
    y_est = linear_function(w_est, xi)
    # Get gradient
    error = y_est - yi
    grad = error * xi
    # Get w
    aux = gamma * grad[:, 0]
    w_est -= aux

print('Estimated w', w_est)
print('Real w', w)
