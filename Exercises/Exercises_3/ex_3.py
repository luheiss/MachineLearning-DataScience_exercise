import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1)- 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

model = make_pipeline(
            StandardScaler(), 
            PolynomialFeatures(degree=2), 
            LinearRegression()
)

model.fit(X, y)

plt.figure(figsize=(12, 6))
plt.scatter(X, y)
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

plt.plot(X_plot, model)
#plt.plot(X_plot, models[2], label=f"Degree {2}")
plt.ylim(y.min() - 1, y.max() + 1)
plt.legend()
plt.show()




model_dt = make_pipeline(
            StandardScaler(),
            DecisionTreeRegressor(max_depth=5)
)   
model_dt.fit(X, y)
y_plot_dt = model_dt.predict(X)
plt.scatter(X, y)
plt.plot(X, y_plot_dt, color='green', label='Decision Tree Regression')
plt.title('Decision Tree Regression Fit (max_depth=5)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()