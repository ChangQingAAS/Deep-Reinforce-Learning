import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

sys.path.append(".")
from args.config import default_params as params


class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer_limit = buffer_limit
        self.buffer = collections.deque(maxlen=self.buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        # print("x.shape is ", x.shape)
        # mu = torch.tensor([1])
        # print("mu.shape is ", mu.shape)
        return mu


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(4, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class OrnsteinUhlenbeckNoise():
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def train_(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, batch_size, gamma, tau):

    s, a, r, s_prime, done_mask = memory.sample(batch_size)
    # print("s_prime.shape is ", s_prime.shape)
    # print(" mu_target(s_prime) is", mu_target(s_prime))

    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(s, mu(s)).mean()  # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


class ddpg_algo():
    def __init__(self):
        super(ddpg_algo, self).__init__()
        self.env = gym.make(params['gym_env'])
        self.buffer_limit = params['buffer_limit']
        self.memory = ReplayBuffer(self.buffer_limit)
        self.q, self.q_target = QNet(), QNet()
        self.q_target.load_state_dict(self.q.state_dict())
        self.mu, self.mu_target = MuNet(), MuNet()
        self.mu_target.load_state_dict(self.mu.state_dict())
        self.lr_mu = params['lr_mu']
        self.lr_q = params['lr_q']
        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=self.lr_mu)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.lr_q)
        self.ou_noise = OrnsteinUhlenbeckNoise(np.zeros(1))
        self.epoch = params['epoch']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.print_interval = params['print_interval']

        self.init_write()

    def init_write(self):
        with open("./result/DDPG.csv", "w+", encoding="utf-8") as f:
            f.write("epoch_number,average reward\n")

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self):
        score = 0.0
        for n_epi in range(self.epoch):
            s = self.env.reset()
            done = False

            while not done:
                a = self.mu(torch.from_numpy(s).float())
                a = a.argmax().item()
                # print("action is ", a)
                s_prime, r, done, info = self.env.step(a)
                self.memory.put((s, a, r, s_prime, done))
                score += r
                s = s_prime

            if self.memory.size() > 2000:
                for i in range(10):
                    train_(self.mu, self.mu_target, self.q, self.q_target, self.memory, self.q_optimizer,
                           self.mu_optimizer, self.batch_size, self.gamma, self.tau)
                    self.soft_update(self.mu, self.mu_target)
                    self.soft_update(self.q, self.q_target)

            if n_epi % self.print_interval == 0:
                with open("./result/DDPG.csv", "a+", encoding="utf-8") as f:
                    f.write("{},{}\n".format(n_epi, score / self.print_interval))
                print("n_episode :{}, score : {:.1f}".format(n_epi, score / self.print_interval))
                score = 0.0

        self.env.close()


if __name__ == '__main__':
    algo = ddpg_algo()
    algo.train()