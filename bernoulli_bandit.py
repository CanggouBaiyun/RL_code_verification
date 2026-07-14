"""Bernoulli multi-armed bandit verification example."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """伯努利多臂老虎机, 输入 K 表示拉杆个数."""

    def __init__(self, K):
        # 随机生成 K 个 0~1 的数, 作为每根拉杆的获奖概率.
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k):
        """选择 k 号拉杆, 返回 1 表示获奖, 返回 0 表示未获奖."""
        if np.random.rand() < self.probs[k]:
            return 1
        return 0


class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = []

    def update_regret(self,k):
        #计算累积懊悔并保存，k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    
    def run_one_step(self):
        #
        raise NotImplementedError
    
    def run(self, num_steps):
        #
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)
        
class EpsilonGreedy(Solver):
    """epsilon贪婪算法，继承Solver类"""
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        #每轮有1%概率探索，99%概率利用当前认为最好的杠杆。
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k) #得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    """epsilon随时间衰减的epsilon-贪婪算法，继承Solver类"""
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        
        return k

class UCB(Solver):
    """UCB算法"""
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB,self).__init__(bandit)
        self.total_count = 0;
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef *np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class ThompsonSampling(Solver):
    """汤普森采样算法"""
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)
    
    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] += (1-r)
        return k


def plot_results(solvers, solver_names):
    """
    生成累积懊悔随时间变化的图像。输入solvers是一个列表，列表中的每个元素是一种特定的策略
    而solver_names也是一个列表，存储每个策略的名称
    """

    #依次遍历每个算法
    for idx,solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


def main():
    
    np.random.seed(1)
    K = 10
    bandit_10_arm = BernoulliBandit(K)

    print("随机生成了一个%d臂伯努利老虎机" % K)
    print(
        "获奖概率最大的拉杆为%d号,其获奖概率为%.4f"
        % (bandit_10_arm.best_idx, bandit_10_arm.best_prob)
    )
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver],["EpsilonGreedy"])
    
    np.random.seed(0)
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [
        EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
    ]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

    np.random.seed(2)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

    np.random.seed(3)
    coef = 1 #控制不确定性比重的系数
    UCB_slover = UCB(bandit_10_arm, coef)
    UCB_slover.run(5000)
    print('上置信界算法的累积懊悔为：',UCB_slover.regret)
    plot_results([UCB_slover],["UCB"])

    np.random.seed(4)
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：',thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver],["ThompsonSampling"])
    
if __name__ == "__main__":
    main()
