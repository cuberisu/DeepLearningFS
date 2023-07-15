import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    return y.astype(int)

x = np.arange(-5.0, 5.0, 0.1)   # -5.0 ~ 5.0 전까지 0.1 간격의 넘파이 배열 생성 
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y축 범위 지정
plt.show()