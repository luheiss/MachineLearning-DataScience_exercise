import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LinearRegression


np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1)- 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.figure(figsize=(12, 6))
plt.scatter(X, y)
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

models = {}

for degree in [ 1, 2, 5,16, 100]:
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    models[degree] = y_plot

plt.plot(X_plot, models[1], label=f"Degree {1}")
plt.plot(X_plot, models[2], label=f"Degree {2}")
plt.plot(X_plot, models[5], label=f"Degree {5}")
plt.plot(X_plot, models[16], label=f"Degree {16}")
plt.plot(X_plot, models[100], label=f"Degree {100}")
plt.ylim(y.min() - 1, y.max() + 1)
plt.legend()
plt.show()



## old code
#poly_kernel_pipeline = make_pipeline(
#    StandardScaler(),
#    PolynomialFeatures(degree=3), 
#    LinearRegression()
#)
#
#poly_kernel_pipeline.fit(X, y)