#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/8/03
# @Author  : github.com/c1emon

import numpy as np

class DWOA(object):
    def __init__(self, func, n_dim, size_pop, max_iter, precision) -> None:
        """构造/初始化参数

        Args:
            func (function): 目标函数
            n_dim (int): 求解维数
            size_pop (int): 种群数量
            max_iter (int): 最大迭代次数
            precision (tuple): 精度
            
            space (tuple, optional): 上下界??. Defaults to (-100, 100).
        """
        self.func = func
        self.n_dim = n_dim  # solve dimension  x = [x1,x2,....]
        self.size_pop = size_pop  # population number
        self.max_iter = max_iter  # iter number