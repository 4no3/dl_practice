# coding: utf-8
import numpy as np


def identity_function(x):   # 恒等関数。回帰モデルの出力層。
    return x


def step_function(x):   # ステップ関数
    # 正の場合 True, 負の場合 False → np.int で True を 1 に、Flase を 0 に変換。
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):     # シグモイド関数
    return 1 / (1 - np.exp(-x))


def sigmoid_grad(x):    # シグモイド関数の解析微分
    # http://nonbiri-tereka.hatenablog.com/entry/2014/06/30/134023
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):    # Relu 関数
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1    # Relu の特性上、x >= 0 で傾きは 1。
    return grad


def softmax(x):     # ソフトマックス関数。分類問題の出力層。
    # 次元について：https://deepage.net/features/numpy-axis.html
    if x.ndim == 2:
        # np.max は axis の方向に因らず、1xn の配列になる。
        # そのため転置しないと次元が一致せず、ブロードキャストできないことがある。
        # 入力を転置して n をそろえて計算してから再転置する。
        # np 配列への慣れと勉強が必要。。。
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):   # 損失関数：二乗和誤差
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):  # 損失関数：交差エントロピー誤差
    if y.dim == 1:  # 1次元の場合、y, t を 1 行 n 列に変換。
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データが one-hot-vector の場合、正解ラベルのインデックスに変換
    # one-hot でなくても、正解ラベルのインデックスになるのでは？という疑問
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):     # ソフトマックス関数の損失計算関数
    y = softmax(X)
    return cross_entropy_error(y, t)
