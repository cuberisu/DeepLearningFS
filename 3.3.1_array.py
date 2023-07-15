import numpy as np
A = np.array([1, 2, 3, 4])
print(A)            # [1 2 3 4]
print(np.ndim(A))   # 1 (배열의 차원 수)
print(A.shape)      # (4,) (배열의 형상(크기)) 1차원이어도 튜플로 반환
print(A.shape[0])   # 4

B = np.array([[1, 2], [3, 4], [5, 6]])
'''
[[1 2]
 [3 4]
 [5 6]]
'''
print(B)
print(np.ndim(B))   # 2
print(B.shape)  # (3, 2)