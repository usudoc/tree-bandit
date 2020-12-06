import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import joblib


# バンディットタスク
class Bandit():
    def __init__(self):
        # バンディットの設定
        self.probability = np.asarray([[0.4, 0.5, 0.6, 0.7],
                                       [0.9, 0.8, 0.1, 0.2],
                                       [0.7, 0.1, 0.3, 0.6],
                                       [0.6, 0.4, 0.1, 0.3],
                                       [0.5, 0.2, 0.3, 0.1]])
        # スタート地点
        self.start = 0
        # ゴール地点
        self.goal = 2

        # 終端状態での確率
        # Q値の更新式に合わせて計算
        self.end_prob = np.asarray([[1.3, 1.2, 0.5, 0.6],
                                    [1.2, 0.6, 0.8, 1.1],
                                    [1.2, 1.0, 0.7, 0.9],
                                    [1.2, 0.9, 1.0, 0.8]])
        self.max_act_prob = 1.3
        # 状態ごとの最もQ値の高い行動
        self.correct_act = np.asarray([[0, 0]])

        """
        for i in range(len(self.probability)):
            for j in range(len(self.probability[0])):
                result = self.probability[0, i] * self.probability[i+1, j]
                # 終端状態の確率を記録する
                self.mul_prob[i, j] = result
                if result > self.max_act_prob:
                    self.max_act_prob = result
                    # [最初の行動, 2回目の行動]
                    self.correct_act[0] = [i, j]
        """

    # 報酬を評価
    def get_reward(self, current_state, before_action, action):
        # 受け取るアクションは0か1の2個
        # あたりなら1を返す
        if self.probability[current_state + before_action, action] >= random.random():
            return 1
        # 外れなら0を返す
        else:
            return 0

    # 後悔の値を返す
    def get_regret(self, before_action, action):
        return self.max_act_prob - self.end_prob[before_action, action]

    # 正しい行動を選んでいるかどうかを返す
    def get_correct(self, current_state, action):
        # 正しければ1、外れていると0を返す
        if self.correct_act[0, current_state] == action:
            return 1
        else:
            return 0

    # 状態の数を返す
    def get_num_state(self):
        return len(self.probability)

    # 行動の数を返す
    def get_num_action(self):
        return len(self.probability[0])

    # スタート地点の場所を返す(初期化用)
    def get_start_state(self):
        return self.start

    # ゴール地点の場所を返す
    def get_goal_state(self):
        return self.goal

    # 行動を受け取り、次状態を返す
    def get_next_state(self, current_state, current_action):
        return current_state + 1


# Q学習のクラス
class Q_learning():
    # 学習率、割引率、状態数、行動数を定義する
    def __init__(self, learning_rate=0.1, discount_rate=0.9, num_state=None, num_action=None):
        self.learning_rate = learning_rate  # 学習率
        self.discount_rate = discount_rate  # 割引率
        self.num_state = num_state  # 状態数
        self.num_action = num_action  # 行動数
        # Qテーブル[num_state+1, num_action]を初期化
        self.Q = np.zeros((self.num_state + 1, self.num_action))
        self.count = np.zeros((5, 4))

    # Q値の更新
    # 現状態、選択した行動、得た報酬、次状態を受け取って更新する
    def update_Q(self, current_state, current_action, reward, next_state, before_action):
        state = current_state + before_action
        # TD誤差の計算
        TD_error = (reward
                    + self.discount_rate
                    * max(self.Q[next_state])
                    - self.Q[state, current_action])
        # Q値の更新
        # print("state ; ", state)
        self.count[state, current_action] += 1
        # print("count", self.count)
        self.learning_rate = 1 / self.count[state, current_action]
        self.Q[state, current_action] += self.learning_rate * TD_error

    # Q値の初期化
    def init_params(self):
        self.Q = np.zeros((self.num_state + 1, self.num_action))
        self.count = np.zeros_like(self.count)

    # Q値を返す
    def get_Q(self):
        return self.Q


# PS
class PS():
    def __init__(self, act_prob=None, k=0):
        self.count = np.zeros(k)
        self.reward = np.zeros(k)
        self.value = np.zeros(k)
        sorted_prob = np.sort(act_prob[0])[::-1]  # バンディットを確率の高い順に要素数のみ変更
        self.r = (sorted_prob[0] + sorted_prob[1]) / 2  # rの設定

    def update_Q(self, current_state, current_action, reward, next_state):
        self.count[current_action] += 1
        self.reward[current_action] += reward
        average = self.reward[current_action] / self.count[current_action]

    def init_params(self):
        pass

    def get_Q(self):
        return self.value


# RS
class RS():
    def __init__(self, act_prob=None, k=0, learning_rate=0.1, discount_rate=1.0, num_state=None, num_action=None):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.num_state = num_state  # 状態数
        self.num_action = num_action  # 行動数
        self.count = np.zeros((5, 4))
        self.reward = np.zeros((5, 4))
        self.value = np.zeros((5, 4))
        sorted_prob = np.sort(act_prob[0])[::-1]  # バンディットを確率の高い順に要素数のみ変更
        self.r = [1.25, 0.85, 0.65, 0.5, 0.4]
        self.t = np.zeros((5, 4))
        self.current_t = np.zeros((5, 4))
        self.post_t = np.zeros((5, 4))
        # Qテーブル[num_state+1, num_action]を初期化
        self.Q = np.zeros((self.num_state + 1, self.num_action))

    def update_Q(self, current_state, current_action, reward, next_state, before_action):
        state = current_state + before_action  # 現在どの場所で行動しようとしているかを記録
        self.count[state, current_action] += 1
        self.reward[state, current_action] += reward
        #average = self.reward[state, current_action] / self.count[state, current_action]

        # TD誤差の計算
        TD_error = (reward
                    + self.discount_rate
                    * max(self.Q[next_state])
                    - self.Q[state, current_action])
        # Q値の更新
        self.Q[state, current_action] += 1 / self.count[state, current_action] * TD_error

        # 信頼度
        self.current_t[state, current_action] += 1
        self.post_t[state, current_action] += 1 / self.current_t[state, current_action]\
                                              * (self.discount_rate * max(self.t[next_state])
                                                 - self.post_t[state, current_action])  # 最大のQ値をとる行動
        self.t[state, current_action] = self.current_t[state, current_action] + self.post_t[state, current_action]

        self.value[state, current_action] = self.t[state, current_action] * (self.Q[state, current_action] - self.r[state])

    def init_params(self):
        self.count = np.zeros_like(self.count)
        self.reward = np.zeros_like(self.reward)
        self.value = np.zeros_like(self.value)
        self.t = np.zeros_like(self.t)
        self.current_t = np.zeros_like(self.current_t)
        self.post_t = np.zeros_like(self.post_t)
        self.Q = np.zeros_like(self.Q)

    def get_Q(self):
        return self.value


# 方策クラス
class Greedy(object):
    # 行動価値を受け取って行動番号を返す
    def select_action(self, value, current_state):
        # return np.argmax(value[current_state])
        max_values = np.where(value[current_state] == value[current_state].max())
        return np.random.choice(max_values[0])

    def init_params(self):
        pass

    def update_params(self):
        pass


# PSのためのGreedy
class Greedy_PS(object):
    def __init__(self, r):
        self.r = r

    # 行動価値を受け取って行動番号を返す
    def select_action(self, value, current_state, k=4):
        if np.max(value[current_state]) > self.r[current_state]:
            max_values = np.where(value[current_state] == value[current_state].max())
            return np.random.choice(max_values[0])
        else:
            return random.randint(0, k-1)

    def init_params(self):
        pass

    def update_params(self):
        pass


# e-greedy
class EpsDecGreedy(object):
    def __init__(self, eps, eps_min, eps_decrease):
        self.eps = eps
        self.eps_init = eps
        self.eps_min = eps_min
        self.eps_decrease = eps_decrease

    def select_action(self, value, current_state, k=4):
        if random.random() < self.eps:
            return random.randint(0, k-1)
        else:
            max_values = np.where(value[current_state] == value[current_state].max())
            return np.random.choice(max_values[0])

    def init_params(self):
        self.eps = self.eps_init

    def update_params(self):
        self.eps -= self.eps_decrease


# エージェントクラス
class Agent():
    def __init__(self, value_func="Q_learning", policy="greedy", learning_rate=0.1, discount_rate=0.9, eps=None,
                 eps_min=None, eps_decrease=None, n_state=None, n_action=None, bandit=None):
        # 価値更新方法の選択
        if value_func == "Q_learning":
            self.value_func = Q_learning(num_state=n_state, num_action=n_action, discount_rate=1.0)

        elif value_func == "RS":
            self.value_func = RS(num_state=n_state, num_action=n_action, act_prob=bandit, k=len(bandit[0]), discount_rate=1.0)

        else:
            print("error:価値関数候補が見つかりませんでした")
            sys.exit()

        # 方策の選択
        if policy == "greedy":
            self.policy = Greedy()

        elif policy == "greedy_ps":
            self.policy = Greedy_PS(r=self.value_func.r)

        elif policy == "eps_greedy":
            self.policy = EpsDecGreedy(eps=eps, eps_min=eps_min, eps_decrease=eps_decrease)

        else:
            print("error:方策候補が見つかりませんでした")
            sys.exit()

    # パラメータ更新(基本呼び出し)
    def update(self, current_state, current_action, reward, next_state, before_action):
        self.value_func.update_Q(current_state, current_action, reward, next_state, before_action)
        self.policy.update_params()

    # 行動選択(基本呼び出し)
    def select_action(self, current_state):
        return self.policy.select_action(self.value_func.get_Q(), current_state)

    # 行動価値の表示
    def print_value(self):
        print(self.value_func.get_Q())

    # 所持パラメータの初期化
    def init_params(self):
        self.value_func.init_params()
        self.policy.init_params()


# メイン関数
def main():
    # ハイパーパラメータ等の設定
    task = Bandit()  # タスク定義

    SIMULATION_TIMES = 10000  # シミュレーション回数
    EPISODE_TIMES = 10000  # エピソード回数

    # エージェントの設定
    agent = {}
    agent[0] = Agent(policy="eps_greedy", eps=1.0, eps_min=0.0, eps_decrease=1.0/2000,
                     n_state=task.get_num_state(), n_action=task.get_num_action())
    agent[1] = Agent(value_func="RS", policy="greedy_ps", bandit=task.probability, n_state=task.get_num_state(), n_action=task.get_num_action())
    agent[2] = Agent(value_func="RS", policy="greedy", bandit=task.probability, n_state=task.get_num_state(), n_action=task.get_num_action())

    accuracy = np.zeros((len(agent), EPISODE_TIMES))
    reward_gragh = np.zeros((len(agent), EPISODE_TIMES))  # グラフ記述用の報酬記録
    regret_gragh = np.zeros((len(agent), EPISODE_TIMES))  # グラフ記述用の報酬記録

    # トレーニング開始
    print("トレーニング開始")
    for simu in range(SIMULATION_TIMES):
        print("simu :", simu)
        for n_agent in range(len(agent)):
            agent[n_agent].init_params()  # エージェントのパラメータを初期化

        for n_agent in range(len(agent)):
            regret = 0.0
            for epi in range(EPISODE_TIMES):
                current_state = task.get_start_state()  # 現在地をスタート地点に初期化
                before_action = 0
                while True:
                    # 行動選択
                    action = agent[n_agent].select_action(current_state + before_action)
                    # 報酬を観測
                    reward = task.get_reward(current_state, before_action, action)
                    reward_gragh[n_agent, epi] += reward
                    # 正確さを観測
                    accuracy[n_agent, epi] += task.get_correct(current_state, action)
                    # 次状態を観測
                    next_state = task.get_next_state(current_state, action)

                    # Q値の更新
                    agent[n_agent].update(current_state, action, reward, next_state, before_action)
                    # ひとつ前の状態での行動を記録
                    before_action = action
                    current_state = next_state
                    # 次状態が終端状態であれば終了
                    if next_state == task.get_goal_state():
                        # 後悔を観測
                        regret += task.get_regret(before_action, action)
                        regret_gragh[n_agent, epi] += regret
                        break
            print(regret)

    print("Q値の表示")
    for n_agent in range(len(agent)):
        agent[n_agent].print_value()

    print("グラフ表示")
    # グラフ書き込み
    """
    plt.plot(reward_gragh[0] / SIMULATION_TIMES, label="Q_learning")
    plt.plot(reward_gragh[1] / SIMULATION_TIMES, label="RS")
    plt.legend()  # 凡例をつける
    plt.title("reward")  # グラフのタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルをつける
    plt.ylabel("sum reward")  # y軸のラベルを付ける
    plt.show()  # グラフを表示
    """
    plt.plot(accuracy[0] / SIMULATION_TIMES, label="e-greedy")
    plt.plot(accuracy[1] / SIMULATION_TIMES, label="PS")
    plt.plot(accuracy[2] / SIMULATION_TIMES, label="RS")
    plt.legend()  # 凡例をつける
    plt.title("accuracy_tree")  # グラフのタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルをつける
    plt.ylabel("accuracy")  # y軸のラベルを付ける
    plt.ylim([0.0, 1.5])
    plt.legend(loc="upper left")
    plt.show()  # グラフを表示

    plt.plot(regret_gragh[0] / SIMULATION_TIMES, label="e-greedy")
    plt.plot(regret_gragh[1] / SIMULATION_TIMES, label="PS")
    plt.plot(regret_gragh[2] / SIMULATION_TIMES, label="RS")
    plt.legend()  # 凡例をつける
    plt.title("regret_tree")  # グラフのタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルをつける
    plt.ylabel("sum regret")  # y軸のラベルを付ける
    plt.legend(loc="upper left")
    plt.show()  # グラフを表示


main()
