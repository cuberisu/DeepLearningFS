import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규 분포로 초기화
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net = simpleNet()
print(net.W)
'''
[[ 1.01814689 -0.58690245 -1.14133821]
 [-0.3852637   0.16303777  0.28513328]]'''

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)    # [ 0.2641508  -0.20540748 -0.42818297]
print(np.argmax(p)) # 최댓값의 인덱스   # 0
t = np.array([0, 0, 1])   # 정답 레이블
print(net.loss(x, t))    # 1.446427578803091

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
'''
[[ 0.2822618   0.17649221 -0.458754  ]
 [ 0.4233927   0.26473831 -0.688131  ]]
'''

# 간단한 함수를 위한 lambda 기법
# f = lambda w: net.loss(x, t)
# dW = numerical_gradient(f, net.W)