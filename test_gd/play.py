import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate the data
visualize = False
verbose = False

w0_t = 2
w1_t = 1
sigma = 1

x_t = np.arange(-100, 101, 0.1)
# x_t = np.arange(0, 10, 1)
noise = sigma * np.random.randn(x_t.size)

y_t = w1_t * x_t + w0_t
y_noise = y_t + noise

if visualize:
    plt.plot(x_t, y_t)
    plt.hold(True)
    plt.plot(x_t, y_noise)
    plt.show()


N = 400  # Number of steps of the simulation
eta = 0.01
w_0 = np.random.rand()
w_1 = np.random.rand()

print ('Random guess about w0 and w1')
print (w_0, w_1)

shuffler = np.random.choice(x_t.size, size=x_t.size, replace=False)
x_data = x_t[shuffler]
y_data = y_noise[shuffler]

# x_data = x_t
# y_data = y_t


def calculate_error(x, y, w0, w1):
    est_y = w0 + x * w1
    diff = est_y - y
    error = np.sum(diff ** 2)

    return np.sqrt(error / x.size)

# This is a cycle
for i in range(N):
    for point in shuffler:
        aux = (w_0 + w_1 * x_data[point] - y_data[point])
        w_0 = w_0 - eta * 2 * aux / x_data.size
        w_1 = w_1 - eta * 2 * x_data[point] * aux / x_data.size
        if verbose:
            print ('Estimated w0 and w1')
            print (w_0, w_1)
    error = calculate_error(x_data, y_data, w_0, w_1)
    print('error = ', error, 'at step = ', i)

# Now we need to calculate the error
print ('Estimated w0 and w1')
print (w_0, w_1)

print ('Truth values of w0 and w1')
print (w0_t, w1_t)
