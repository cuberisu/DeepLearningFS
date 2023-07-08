# 1.5 넘파이

# 1.5.1 넘파이 가져오기
import numpy as np  # 외부 라이브러리인 numpy를 np라는 이름으로 가져오기


# 1.5.2 넘파이 배열 만들기: np.array()
# 파이썬의 리스트를 인수로 받아 특수한 형태의 배열을 반환한다.
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))


# 1.5.3 넘파이의 산술 연산
# 배열 x와 y는 원소 수가 같다.
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)
print(x / y)

# 브로드캐스트
# 스칼라값과 넘파이 배열의 산술 연산
print(x / 2.0)


# 1.5.4 넘파이의 N차원 배열
A = np.array([[1, 2], [3, 4]])  # 2차원 배열
print(A)
print(A.shape)  # 행렬의 형상
print(A.dtype)  # 행렬에 담긴 원소의 자료형

# 다차원 배열의 산술 연산
# 산술 연산은 형상(차원의 크기, 원소 수)이 같은 행렬끼리 가능
B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)

# 브로드캐스트
print(A*10)


# 1.5.5 브로드캐스트
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)


# 1.5.6 원소 접근
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])     # 0행
print(X[0][1])  # (0, 1) 위치의 원소

for row in X:   # for문으로도 각 원소에 접근할 수 있다.
    print(row)
    
X = X.flatten() # X를 1차원 배열로 변환 (평탄화)
print(X)
print(X[np.array([0, 2, 4])])   # 0, 2, 4, 인덱스의 원소 얻기
print(X > 15)   # bool 배열
print(X[X>15])  # True에 해당하는 원소만
