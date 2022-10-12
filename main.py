import numpy as np
import math

import matplotlib.pyplot as plt



path_A = "/Users/abhinabacharya/PycharmProjects/Quantitative-Foundations-Project-2/input/fun2_A.txt"
path_b = "/Users/abhinabacharya/PycharmProjects/Quantitative-Foundations-Project-2/input/fun2_b.txt"
path_c = "/Users/abhinabacharya/PycharmProjects/Quantitative-Foundations-Project-2/input/fun2_c.txt"

A = np.loadtxt(path_A, )
A = np.reshape(A, (500, 100))
b = np.loadtxt(path_b, )
b = b.reshape((b.shape[0], 1))
c = np.loadtxt(path_c, )
c = c.reshape((c.shape[0], 1))


def fun1(X):
    sum = 0.0
    for i in range(100):
        sum += (i + 1) * float(X[i]) * float(X[i])
    return sum


def gfun1(X):
    g = []
    for i in range(100):
        g.append(float(2) * float(i + 1) * float(X[i]))
    return np.reshape(np.array(g), (X.shape[0], 1))


def g2fun1(X):
    g = []
    for i in range(X.shape[0]):
        g.append(float(2) * float(i + 1))
    return np.reshape(np.array(g), (X.shape[0], 1))


def hess1(x):
    hessian = np.zeros((100, 100), dtype=np.float32)
    for i in range(100):
        hessian[i][i] = 2 * (i + 1)
    return hessian


def fun2(Xi):
    t_diff = b - np.matmul(A, Xi)
    # A
    adj_X = []
    for x in t_diff:
        if x < 0:
            # adj_X.append(1)
            adj_X.append(x)
        else:
            adj_X.append(x)
    adj_X = np.array(adj_X, dtype=float).reshape((len(adj_X), 1))

    return np.matmul(c.transpose(), Xi) - np.sum(np.log(adj_X))


def gfun2(Xi):
    den = (b - np.matmul(A, Xi))
    return c + np.matmul(A.transpose(), (1 / den))


def fun3(Xi):
    return (1 - Xi[0]) ** 2 + 100 * (Xi[1] - Xi[0] ** 2) ** 2


def gfun3(xi):
    g = []
    g.append(-2 + 2 * xi[0] - 400 * (xi[1] - xi[0] ** 2) * xi[0])
    g.append(200 * (xi[1] - xi[0] ** 2))
    return np.array(g)


def backtrack(x0, f, dfx1, dfx2, t, alpha, beta, count):
    com_grad = np.array([dfx1(x0), dfx2(x0)])
    # com_grad = np.reshape(com_grad, (com_grad.shape[0], 1))

    while (f(x0) - (f(x0 - t*com_grad) + alpha * t * np.dot(com_grad, com_grad))) < 0:
        t *= beta
        print("iteration {}").format(count)
        print("Inequality: ",  f(x0) - (f(x0 - t*com_grad) + alpha * t * np.dot(com_grad, com_grad)))
        count += 1
    return t


def line_search(x, a, p, f, gf, c, r):
    max_iter = 1000
    for i in range(max_iter):
        t = a * p
        lhs = f(x + t)
        rhs = f(x) + c * a * np.matmul(p.transpose(), gf(x))
        if lhs <= rhs[0][0]:
            # print("satisfy", i, a)
            return a
        else:
            a = r * a
    return a


def gradient_descent(func):
    if func == 1:
        max_iter = 20
        Xi = np.ones((100, 1))
        # Xi = -np.ones((100, 1))*0.5
        alpha = 0.001
        ap = 0.01
        f_values = []
        x_values = []
        for i in range(max_iter):
            p = -gfun1(Xi)
            ap = line_search(Xi, ap, p, fun1, gfun1, 0.2, 0.99)
            Xi = Xi + ap * p
            f_values.append(fun1(Xi))
            # x_values.append(Xi[0][0])
        print("min_val: ", fun1(Xi), ap)

        # plt.plot(x_values, f_values)
        plt.plot(f_values)
        plt.ylabel('Function 1')
        plt.xlabel('Iteration')

        plt.savefig('/Users/abhinabacharya/PycharmProjects/Quantitative-Foundations-Project-2/plots/gd_fun1.png',
                    format='png', dpi=300,
                    bbox_inches='tight')

        plt.show()
        plt.close()

    if func == 2:
        max_iter = 1000
        # Xi = -np.ones((100, 1))*0.4
        Xi = np.zeros((100, 1))
        alpha = 0.001
        ap = 1
        f_values = []
        x_values = []
        for i in range(max_iter):
            p = -gfun2(Xi)
            if np.sum(b < np.matmul(A, Xi)) > 0:
                ap = ap * 0.99
            else:
                ap = line_search(Xi, ap, p, fun2, gfun2, 0.5, 0.99)
            Xi = Xi + ap * p
            f_values.append(fun2(Xi)[0][0])
            # x_values.append(Xi[0][0])
        print("min_val: ", fun2(Xi), ap)

        # plt.plot(x_values, f_values)
        plt.plot(f_values)
        plt.ylabel('Function 2')
        plt.xlabel('Iteration')

        plt.savefig('/Users/abhinabacharya/PycharmProjects/Quantitative-Foundations-Project-2/plots/gd_fun2.png',
                    format='png', dpi=300,
                    bbox_inches='tight')

        plt.show()
        plt.close()


x_ini = np.ones([100, 1]) * 0.9
alpha = 1
rho = 0.99
slope = 0.5

p = -gfun1(x_ini)
# new_alp = line_search(x_ini, alpha, p, fun1, gfun1, slope, rho)
gradient_descent(2)
pass
