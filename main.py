import numpy as np
import random
import math
import dwoa
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False

def self_fun(x):
    y = (4 - 2.1 * (x[0] ** 2) + (x[0] ** 4) / 3) * (x[0] ** 2) + x[1] * x[0] + (-4 + 4 * (x[1] ** 2)) * (x[1] ** 2)
    return y

if __name__ == '__main__':
    cir = 30
    Y_ax = []
    X_ax = np.arange(0, cir, 1)
    best_list = []
    F = 0
    for i in range(cir):
        model = WOA(D=10, size=30, sum_iter=1000, space=(-100, 100), function=self_fun)
        best, fmin = model.run()
        Y_ax.append(fmin)
        best_list.append(best)
        print('-------------------------------------------------------------')
        print('BEST=', np.around(best, decimals=5), 'min of fitness=', fmin)
    plt.plot(X_ax, Y_ax, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='function value')
    plt.legend(loc="upper right")
    plt.ylabel('函数值')
    plt.xlabel('迭代次数')
    plt.show()
    print('最优值:', min(Y_ax), best_list[Y_ax.index(min(Y_ax))])
    print('最差值:', max(Y_ax))
    current_best = min(Y_ax)
    ave = sum(Y_ax) / cir
    print('平均值:', ave)
    a = np.array([ave] * cir)
    b = np.array(Y_ax)
    std = np.sqrt(sum(np.square(a - b)) / 100)
    print('标准差:', std)