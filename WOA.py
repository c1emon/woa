#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/8/03
# @Author  : github.com/c1emon

import numpy as np

class WOA(object):
    def __init__(self, func, n_dim, size_agent, max_iter, lb, ub, verbose=False, **kwargs) -> None:
        """构造/初始化参数

        Args:
            func (function): 目标函数
            n_dim (int): 求解维数
            size_agent (int): 种群数量
            max_iter (int): 最大迭代次数
            lb (array_like): 变量下界
            ub (array_like): 变量上界
        """
        self.func = func
        self.n_dim = n_dim  # solve dimension
        self.size_agent = size_agent  # agent number
        self.max_iter = max_iter  # max iter number
        
        # init all agents
        self.agents = np.zeros((self.size_agent, self.n_dim))
        
        # boundary
        self.Lb = lb
        self.Ub = ub

        # fitness of every agent
        self.fitness = np.zeros(self.size_agent)
        # best agent and it's fitness
        self.best_agent = np.zeros(self.n_dim)
        self.fitness_min = 0.
        
    def _init_vlaues(self):
        # random init all agents
        for i in range(self.size_agent):
            self.agents[i, :] = self.Lb + (self.Ub - self.Lb) * np.random.uniform(0, 1, self.n_dim)
            self.fitness[i] = self.func(self.agents[i, :])
        # find the best agent at the initial time
        self.fitness_min = np.min(self.fitness)
        self.best_agent = self.agents[np.argmin(self.fitness), :]
        
    def _clip_agent(self, agent):
        clipped = np.zeros(self.n_dim)
        for i in range(self.n_dim):
            x = agent[i]
            lb = self.Lb[i]
            ub = self.Ub[i]
            x = lb if lb > x else x
            x = ub if ub < x else x
            clipped[i] = x
        return clipped
        
    def run(self):
        self._init_vlaues()
        for step in range(self.max_iter):
            a1 = 2. - 2. * (step/self.max_iter)
            a2 = (-1.) + (-1.) * (step/self.max_iter)
            # for each agent
            for i in range(self.size_agent):
                # random numbers
                r1 = np.random.uniform(0, 1, self.n_dim)
                r2 = np.random.uniform(0, 1, self.n_dim)
                # coefficients
                A = 2. * a1 * r1 - a1 # Uniform distribution in [-a, a)
                C = 2. * r2 #  Uniform distribution in [0, 2)
                b = 1. # unknown, default set to 1
                l = (a2 - 1) * np.random.uniform(0, 1) + 1 # Uniform distribution in [-1, 1)
                
                if np.random.random() >= 0.5:
                    # 狩猎
                    D_prime = np.abs(self.best_agent - self.agents[i])
                    agent = D_prime  * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_agent
                else:
                    # 包围
                    if np.linalg.norm(A) < 1:
                        # |A| < 1
                        D = np.abs(C * self.best_agent - self.agents[i])
                        agent = self.best_agent - A * D
                    else:
                        random_agent = np.random.randint(0, self.size_agent - 1)
                        D_rand = np.abs(C * self.agents[random_agent] - self.agents[i])
                        agent = self.agents[random_agent] - A * D_rand
                
                # check if this agent out of boundary, if out amend it
                agent = self._clip_agent(agent)
                fitness = self.func(agent)
                # update this agent
                self.agents[i, :] = agent
                self.fitness[i] = fitness
            
            # find the best at this step
            self.fitness_min = np.min(self.fitness)
            self.best_agent = self.agents[np.argmin(self.fitness), :]
        return self.best_agent, self.fitness_min