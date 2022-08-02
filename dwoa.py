import numpy as np
import random
import math

class WOA(object):
    def __init__(self, D, size, sum_iter, function, space=(-100, 100)):
        self.D = D  # solve dimension  x = [x1,x2,....]
        self.size = size  # population number
        self.iter = sum_iter  # iter number
        self.spaceUP = space[1]
        self.spaceLOW = space[0]
        self.fun = function

        self.X = np.zeros((self.size, self.D))  # space sovle
        self.Lb = self.spaceLOW * np.ones(self.D)
        self.Ub = self.spaceUP * np.ones(self.D)

        self.fitness = np.zeros(self.size)  # 个体适应度
        self.best = np.zeros(self.D)  # 最好的solution
        self.fmin = 0.0

    def init_solve(self):
        for i in range(self.size):
            self.X[i] = self.Lb + (self.Ub - self.Lb) * np.random.uniform(0, 1, self.D)
            self.fitness[i] = self.fun(self.X[i])
        self.fmin = np.min(self.fitness)
        fmin_arg = np.argmin(self.fitness)
        self.best = self.X[fmin_arg]

    def limit(self, before_X):
        for i in range(self.D):
            if before_X[i] < self.spaceLOW:
                before_X[i] = self.spaceLOW
            elif before_X[i] > self.spaceUP:
                before_X[i] = self.spaceUP
        return before_X

    def run(self):
        self.init_solve()
        ka = 0
        kr = 0
        for step in range(self.iter):
            a = 2 - step * (2 / self.iter)  # 线性下降权重2 - 0
            # a = 1.5 - 1.2/(1+np.exp(-20*(2*step-self.iter)/(2*self.iter)))   # 改进
            a2 = -1 + step * (-1 / self.iter)  # 线性下降权重-1 - -2
            for i in range(self.size):
                r1 = np.random.uniform(0, 1, self.D)
                r2 = np.random.uniform(0, 1, self.D)
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = (a2 - 1) * random.random() + 1   # 原文为[-1,1]的随机数
                p = np.random.uniform(0, 1)
                if p > 0.5:
                    S = self.best + abs(self.best - self.X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l)
                else:
                    if abs(np.random.choice(A, 1)[0]) < 1:     # 对应于 |A|<1 原论文A为向量，计算A的模则全部小于1，因此改为随机选择的方式。
                        S = self.best - abs(C * self.best - self.X[i]) * A
                    else:
                        temp = np.random.randint(0, self.size - 1)
                        S = self.X[temp] - abs(C * self.X[temp] - self.X[i]) * A
                S= self.limit(S)
                Fnew = self.fun(S)
                if Fnew < self.fitness[i]:
                    self.X[i] = S
                    self.fitness[i] = Fnew
            self.fmin = np.min(self.fitness)
            fmin_arg = np.argmin(self.fitness)
            self.best = self.X[fmin_arg]
        return self.best, self.fmin
