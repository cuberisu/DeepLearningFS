import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))
'''
[[19 22]
 [43 50]]
'''

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)  # (2, 3)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)  # (3, 2)

print(np.dot(A, B))
'''
[[22 28]
 [49 64]]
'''