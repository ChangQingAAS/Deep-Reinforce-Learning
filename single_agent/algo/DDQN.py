import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
import gym
import sys

sys.path.append(".")
from args.config import ddqn_params as params


class Net(nn.Module):

    def __init__(self, STATE_NUM, ACTION_NUM):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=STATE_NUM, out_features=128)  # 输入层
        self.fc2 = nn.Linear(in_features=128, out_features=ACTION_NUM)  # 输出层

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.fc2(x)
        return action_value


class DDQN_ALGO(object):

    def __init__(self, path):
        self.env = gym.make(params['gym_env'])
        self.GAMMA = params['gamma']
        self.learning_rate = params['learning_rate']
        self.epoch = params['epoch']
        self.BATCH_SIZE = params['batch_size']
        self.EPSILON = params['epsilon']
        self.TARGET_REPLACE_ITER = params['target_replace_iter']
        self.MEMORY_CAPACITY = params['memory_capacity']
        self.print_interval = params['print_interval']
        self.train_number = params['train_number']
        self.path = path

        self.position = 0

        self.ACTION_NUM = self.env.action_space.n
        self.STATE_NUM = self.env.observation_space.shape[0]
        self.ENV_A_SHAPE = 0 if isinstance(self.env.action_space.sample(), int) else env.action_space.sample().shape

        self.eval_net = Net(self.STATE_NUM, self.ACTION_NUM)
        self.target_net = Net(self.STATE_NUM, self.ACTION_NUM)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.learn_step_counter = 0

        self.loss_func = nn.MSELoss()

        self.memory = np.zeros((self.MEMORY_CAPACITY, self.STATE_NUM * 2 + 2))

        self.init_write()

    def init_write(self):
        for i in range(self.train_number):
            with open(self.path + "/result/DDQN/result_%s.csv" % str(i), "w+", encoding="utf-8") as f:
                f.write("epoch_number,average reward\n")

    def choose_action(self, x):  # epsilon-greedy策略：避免收敛在局部最优
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() <= self.EPSILON:  # 以epsilon的概率利用
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            if self.ENV_A_SHAPE == 0:
                action = action[0]
            else:
                action = action.reshape(self.ENV_A_SHAPE)
        else:  # 以1-epsilon的概率探索
            action = np.random.randint(0, self.ACTION_NUM)
            if self.ENV_A_SHAPE == 0:
                action = action
            else:
                action = action.reshape(self.ENV_A_SHAPE)
        return action

    def save_transition(self, state, action, next_state, reward):  # 将转移存储在记忆池中
        transition = np.hstack((state, [action, reward], next_state))  # 将参数平铺在数组之中
        self.memory[self.position % self.MEMORY_CAPACITY, :] = transition
        self.position += 1

    # * 学习
    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            # TargetNet: 用eval_net来更改targetnet的参数
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)  # 在memory中抽样: memory 必然是满的，才会进入到learn
        batch_memory = self.memory[sample_index, :]

        batch_state = torch.FloatTensor(batch_memory[:, :self.STATE_NUM])
        batch_action = torch.LongTensor(batch_memory[:, self.STATE_NUM:self.STATE_NUM +
                                                     1].astype(int))  # 对应的transition中的action
        batch_reward = torch.FloatTensor(batch_memory[:,
                                                      self.STATE_NUM + 1:self.STATE_NUM + 2])  # 对应的transition中的reward
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.STATE_NUM:])  # 对应的是最后的next_state的部分

        q_eval = self.eval_net(batch_state).gather(1, batch_action)  # 从eval中获取价值函数
        q_next = self.target_net(batch_next_state).detach()  # 切一段下来，避免反向传播
        q_target = batch_reward + self.GAMMA * \
            q_next.max(1)[0].view(self.BATCH_SIZE, 1)  # 使用target_net来推荐最大reward值

        loss = self.loss_func(q_eval, q_target)  # 计算loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for train_counter in range(self.train_number):
            total_reward = 0
            for n_epi in range(self.epoch):
                state = self.env.reset()
                done = False
                while not done:

                    action = self.choose_action(state)  # 选择action epsilon-greedy 策略
                    next_state, reward, is_done, _ = self.env.step(action)  # 与环境交互

                    x, x_dot, theta, theta_dot = next_state
                    reward1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                    reward2 = (self.env.theta_threshold_radians - abs(theta)) / \
                        self.env.theta_threshold_radians - 0.5
                    reward = reward1 + reward2  # 计算reward

                    self.save_transition(state, action, next_state, reward)  # 存储转移
                    state = next_state  # 继续新状态下的前进

                    total_reward += reward
                    if is_done:
                        break

                if self.position > self.MEMORY_CAPACITY:  # 仅有memory满载时才学习
                    self.learn()

                if n_epi % self.print_interval == 0:
                    with open(self.path + "/result/DDQN/result_%s.csv" % str(train_counter), "a+",
                              encoding="utf-8") as f:
                        f.write("{},{}\n".format(n_epi, total_reward / self.print_interval))
                    print("episode :{}, average  score : {}".format(n_epi, total_reward / self.print_interval))
                    total_reward = 0.0
            self.env.close()


if __name__ == "__main__":
    path = sys.path[0].rsplit("/", 1)[0]
    algo = DDQN_ALGO(path)
    algo.train()
