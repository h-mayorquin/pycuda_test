import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler

# Generate the data
visualize = False
verbose = False

w0_t = 0
w1_t = 1
sigma = 10

x_t = np.arange(-100, 101, 0.1)
noise = sigma * np.random.randn(x_t.size)

y_t = w1_t * x_t + w0_t
y_noise = y_t + noise

# Now here we do the classification
alpha = 0.0001
average= False
epsilon = 0.1
eta= 0.01
fit_intercept = True
n_iter = 100

scaler = StandardScaler()
scaler.fit(x_t)
aux_std = x_t.std()
aux_mean = x_t.mean()

x_t = scaler.transform(x_t)

X = np.vstack((np.ones_like(x_t), x_t)).T

reg = lm.SGDRegressor(n_iter=n_iter, fit_intercept=fit_intercept)
reg.fit(X, y_t)
coeff = reg.coef_
print('Unscaled coefficients', coeff)
coeff_scaled = coeff[1] * aux_std + aux_mean
coeff_scaled = coeff[1] / aux_std
coeff_scaled_2 = coeff[0] - aux_mean
print('Scaled coefficients', coeff_scaled, coeff_scaled_2)

scaler.fit(x_t)
