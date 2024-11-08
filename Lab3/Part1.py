import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5) # Generate isotropic Gaussian blobs for clustering.
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu') # Create a scatter plot of y vs X.
plt.title('Generated Data')
#plt.show()

from sklearn.naive_bayes import GaussianNB
model = GaussianNB() # Create a Gaussian Naive Bayes classifier.
model.fit(X, y) # Fit Gaussian Naive Bayes according to X, y.

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2) # Return a sample (or samples) from the “standard normal” distribution.
ynew = model.predict(Xnew) # Perform classification on an array of test vectors Xnew.

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu', alpha=0.5) # Create a scatter plot of y vs X.
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1) # Create a scatter plot of ynew vs Xnew.
plt.axis(lim)
plt.title('Generated Data with Predicted Values')
plt.show()

yprob = model.predict_proba(Xnew) # Return probability estimates for the test vector Xnew.
print(yprob[-8:].round(2)) # Print the last 8 values of yprob rounded to 2 decimal places.

# How does the Gaussian Naive Bayes classifier work?
# The Gaussian Naive Bayes classifier assumes that the likelihood of the features is Gaussian:
# P(x_i|y) = (1 / sqrt(2 * pi * sigma_y^2)) * exp(- (x_i - mu_y)^2 / (2 * sigma_y^2))
# where mu_y and sigma_y are the mean and variance of the feature x_i for class y.
# The likelihood of the features is assumed to be independent given the class:
# P(x|y) = P(x_1|y) * P(x_2|y) * ... * P(x_n|y)
# The prior probability of the class is estimated from the training data:
# P(y) = (number of samples in class y) / (total number of samples)
# The posterior probability of the class given the features is estimated using Bayes' theorem:
# P(y|x) = P(x|y) * P(y) / P(x)
# The class with the highest posterior probability is the predicted class: