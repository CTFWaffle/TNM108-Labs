import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import numpy as np

# Simple Linear Regression

rng = np.random.RandomState(1)
x = 10 * rng.rand(50) # 50 random numbers between 0 and 10
y = 2 * x - 5 + rng.randn(50) # y = 2x - 5 + noise
plt.scatter(x, y) # Scatter plot
plt.title('Simple Linear Regression without Line')
plt.show()


from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y) # Fit the model
xfit = np.linspace(0, 10, 1000) # 1000 points between 0 and 10
yfit = model.predict(xfit[:, np.newaxis]) # Predict y values based on x values
plt.scatter(x, y) # Scatter plot
plt.plot(xfit, yfit) # Line plot   
plt.title('Simple Linear Regression')
plt.show()

#print("Model slope: ", model.coef_[0])
#print("Model intercept: ", model.intercept_)




# Polynomial Regression
# y = a + bx + cx^2 + dx^3 + ...
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3) # 100 random numbers between 0 and 10
Y = 0.5 + np.dot(X, [1.5, -2., 1.]) # y = 0.5 + 1.5x - 2x^2 + x^3 + noise
model.fit(X, Y) # Fit the model
#print(model.intercept_)
#print(model.coef_)

from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4]) # 3 numbers
poly = PolynomialFeatures(3, include_bias=False) # 3rd degree polynomial
#print(poly.fit_transform(x[:, None])) # x, x^2, x^3

# Polynomial Regression with Pipeline (Allowing for more complex relationships between x and y)
from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression()) # 7th degree polynomial
rng = np.random.RandomState(1)
x = 10 * rng.rand(50) # 50 random numbers between 0 and 10
y = np.sin(x) + 0.1 * rng.randn(50) # y = sin(x) + noise
xfit = np.linspace(0, 10, 1000) # 1000 points between 0 and 10
poly_model.fit(x[:, np.newaxis], y) # Fit the model
yfit = poly_model.predict(xfit[:, np.newaxis]) # Predict y values based on x values
plt.scatter(x, y) # Scatter plot
plt.plot(xfit, yfit) # Line plot
plt.title('Polynomial Regression')
plt.show()

# Gaussian Basis Functions
from sklearn.base import BaseEstimator, TransformerMixin
class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""

    def __init__(self, N, width_factor=2.0): 
        self.N = N
        self.width_factor = width_factor

    @staticmethod 
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
    
    def fit(self, X, y=None): 
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
    
    def transform(self, X): 
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, 
                                 self.width_, axis=1)
    
gauss_model = make_pipeline(GaussianFeatures(20),LinearRegression()) # 20 Gaussian basis functions
gauss_model.fit(x[:, np.newaxis], y) # Fit the model
yfit = gauss_model.predict(xfit[:, np.newaxis]) # Predict y values based on x values

plt.scatter(x, y) # Scatter plot
plt.plot(xfit, yfit) # Line plot
plt.xlim(0, 10) # x-axis limits
plt.title('Gaussian Basis Functions')
plt.show()

# Regularization (Preventing overfitting)

# First overfitting example
model = make_pipeline(GaussianFeatures(30), LinearRegression()) # Gaussian basis function increased by 10 resulting in overfitting
model.fit(x[:, np.newaxis], y)
plt.scatter(x, y) # Scatter plot
plt.plot(xfit, model.predict(xfit[:, np.newaxis])) # Line plot
plt.xlim(0, 10) # x-axis limits
plt.ylim(-1.5, 1.5) # y-axis limits
plt.title('Overfitting Example')
plt.show()

# Plotting the coefficients of the Gaussian bases to show overfitting
def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    
    if title:
        ax[0].set_title(title)
    ax[1].plot(model.steps[0][1].centers_, model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location', ylabel='coefficient', xlim=(0, 10))

    plt.show()
model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model, title='Overfitting Example with Coefficients')

# Ridge Regression (L2 Regularization)
from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1)) # alpha is the regularization strength,
# (higher alpha = more regularization(less overfitting, by penalizing large coefficients))   
basis_plot(model, title='Ridge Regression with alpha=0.1')

# Ridge Regression with varied alpha values
'''
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.001))
basis_plot(model, title='Ridge Regression with alpha=0.001')
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.5))
basis_plot(model, title='Ridge Regression with alpha=0.5')
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.9))
basis_plot(model, title='Ridge Regression with alpha=0.9')
'''
# Conclusion for Ridge Regression with varied alpha values
#High alpha values result in more regularization (less overfitting)
#Low alpha values result in less regularization (more overfitting)
#Very high alpha values result in underfitting, very low alpha values result in no penalty for large coefficients (i.e overfitting)


# Lasso Regression (L1 Regularization)
from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001)) # alpha is still the regularization strength
basis_plot(model, title='Lasso Regression')

# Lasso Regression with varied alpha values
'''
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.01))
basis_plot(model, title='Lasso Regression with alpha=0.01')
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.005))
basis_plot(model, title='Lasso Regression with alpha=0.005')
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.009))
basis_plot(model, title='Lasso Regression with alpha=0.009')
'''
# Conclusion for Lasso Regression with varied alpha values
# Just like Ridge Regression, high alpha values result in more regularization (less overfitting) and so on
# Lasso Regression is much more sensitive to alpha values than Ridge Regression, needing far smaller alpha values.


