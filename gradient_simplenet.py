# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # ガウス分布で重みを初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)  # 分類問題の出力層
        loss = cross_entropy_error(y, t)
        return loss


net = simpleNet()
print(net.W)    # 重みパラメータ

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p))    # 最大値のインデックス

t = np.array([0, 0, 1])     # 正解ラベル
loss = net.loss(x, t)
print(loss)


def f(W):
    return net.loss(x, t)
# f = lambda w: net.loss(x, t) でも同じだけど、pep8 では推奨されないみたい。


dW = numerical_gradient(f, net.W)
print(dW)
