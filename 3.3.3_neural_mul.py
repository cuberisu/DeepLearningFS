import numpy as np
X = np.array([1, 2])
print(X.shape)  # (2,)

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W.shape)  # (2, 3)

Y = np.dot(X, W)
print(Y)    # [ 5 11 17]
