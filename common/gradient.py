# coding: utf-8
import numpy as np


def numerical_gradient_1d(f, x):
    # 中心差分で微分する
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val    # 値を元に戻す
    return grad


def numerical_gradient_2d(f, X):
    if X.dim == 1:
        # 関数の前の _ の意味
        # →https://qiita.com/ikki8412/items/ab482690170a3eac8c76
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        return grad


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # イテレータを生成している模様。多次元の入力への対応。
    # 詳しい解説はそのうち学ぼう。
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)     # f(x+h)

        x[idx] = tmp_val - h
        tmp_val = x[idx]
        fxh2 = f(x)     # f(x - h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val    # 値を元に戻す
        it.iternext()

    return grad
