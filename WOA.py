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
            X = lb if lb > x else x
            X = ub if ub < x else x
            clipped[i] = X
        return clipped
        
    def run(self):
        self._init_vlaues()
        for step in range(self.max_iter):
            a = 2 - 2 * (step/self.max_iter)
            a2 = -1 + -1 * (step/self.max_iter)
            # each agent
            for i in range(self.size_agent):
                # r1 r2有待考证
                r1 = np.random.uniform(0, 1, self.n_dim)
                r2 = np.random.uniform(0, 1, self.n_dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = (a2 - 1) * np.random.random() + 1
                p = np.random.uniform(0, 1)
                
                if np.random.uniform(0, 1) > 0.5:
                    agent = self.best_agent + np.abs(self.best_agent - self.agents[i]) * np.exp(b * l) * np.cos(2 * np.pi * l)
                else:
                    
                    if np.linalg.norm(A) < 1:
                        # |A| < 1
                        agent = self.best_agent - np.abs(C * self.best_agent - self.agents[i]) * A
                    else:
                        temp = np.random.randint(0, self.size_agent - 1)
                        agent = self.agents[temp] - np.abs(C * self.agents[temp] - self.agents[i]) * A
                
                agent = self._clip_agent(agent)
                fitness = self.func(agent)
                if fitness < self.fitness[i]:
                    # agent at now is best, so update
                    self.agents[i, :] = agent
                    self.fitness[i] = fitness
            # find the best at this step
            self.fitness_min = np.min(self.fitness)
            self.best_agent = self.agents[np.argmin(self.fitness), :]
        return self.best_agent, self.fitness_min