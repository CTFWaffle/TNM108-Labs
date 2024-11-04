import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# Simple Linear Regression

rng = np.random.RandomState(1)
x = 10 * rng.rand(50) # 50 random numbers between 0 and 10
y = 2 * x - 5 + rng.randn(50) # y = 2x - 5 + noise
plt.scatter(x, y) # Scatter plot
plt.show()


from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y) # Fit the model
xfit = np.linspace(0, 10, 1000) # 1000 points between 0 and 10
yfit = model.predict(xfit[:, np.newaxis]) # Predict y values based on x values
plt.scatter(x, y) # Scatter plot
plt.plot(xfit, yfit) # Line plot   
plt.show()

#print("Model slope: ", model.coef_[0])
#print("Model intercept: ", model.intercept_)




# Polynomial Regression
# y = a + bx + cx^2 + dx^3 + ...
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3) # 100 random numbers between 0 and 10
Y = 0.5 + np.dot(X, [1.5, -2., 1.]) # y = 0.5 + 1.5x - 2x^2 + x^3 + noise
model.fit(X, Y) # Fit the model
print(model.intercept_)
print(model.coef_)