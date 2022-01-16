import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random
import sys

sys.path.append(".")
from args.config import sac_params as params


class ReplayBuffer():

    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

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


class PolicyNet(nn.Module):

    def __init__(self, learning_rate, lr_alpha, init_alpha, target_entropy, in_dim):
        self.target_entropy = target_entropy
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class QNet(nn.Module):

    def __init__(self, learning_rate, tau, in_dim):
        super(QNet, self).__init__()
        self.tau = tau
        self.fc_s = nn.Linear(in_dim, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


def calc_target(pi, q1, q2, mini_batch, gamma):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target


class sac_algo():

    def __init__(self, path):
        super(sac_algo, self).__init__()
        self.path = path
        self.env = gym.make(params['gym_env'])
        self.lr_q = params['lr_q']
        self.lr_pi = params['lr_pi']
        self.epoch = params['epoch']
        self.print_interval = params['print_interval']
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.tau = params['tau']
        self.lr_alpha = params['init_alpha']
        self.init_aplha = params['init_alpha']
        self.target_entropy = params['target_entropy']
        self.buffer_limit = params['buffer_limit']
        self.train_number = params['train_number']
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.memory = ReplayBuffer(self.buffer_limit)
        self.q1 = QNet(self.lr_q, self.tau, self.obs_dim)
        self.q1_target = QNet(self.lr_q, self.tau, self.obs_dim)
        self.q2 = QNet(self.lr_q, self.tau, self.obs_dim)
        self.q2_target = QNet(self.lr_q, self.tau, self.obs_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.pi = PolicyNet(self.lr_pi, self.lr_alpha, self.init_aplha, self.target_entropy, self.obs_dim)

        self.init_write()

    def init_write(self):
        for i in range(self.train_number):
            with open(self.path + "/result/SAC/result_%s.csv" % str(i), "w+", encoding="utf-8") as f:
                f.write("epoch_number,average reward\n")

    def train(self):
        for train_counter in range(self.train_number):
            for n_epi in range(self.epoch):
                score = 0.0
                s = self.env.reset()
                done = False

                while not done:
                    a, log_prob = self.pi(torch.from_numpy(s).float())
                    s_prime, r, done, info = self.env.step([2.0 * a.item()])
                    self.memory.put((s, a.item(), r / 10.0, s_prime, done))
                    score += r
                    s = s_prime

                if self.memory.size() > 1000:
                    for i in range(20):
                        mini_batch = self.memory.sample(self.batch_size)
                        td_target = calc_target(self.pi, self.q1_target, self.q2_target, mini_batch, self.gamma)
                        self.q1.train_net(td_target, mini_batch)
                        self.q2.train_net(td_target, mini_batch)
                        entropy = self.pi.train_net(self.q1, self.q2, mini_batch)
                        self.q1.soft_update(self.q1_target)
                        self.q2.soft_update(self.q2_target)

                if n_epi % self.print_interval == 0:
                    with open(self.path + "/result/SAC/result_%s.csv" % str(train_counter), "a+",
                              encoding="utf-8") as f:
                        f.write("{},{}\n".format(n_epi, score / self.print_interval))
                    print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(
                        n_epi, score / self.print_interval, self.pi.log_alpha.exp()))
                    score = 0.0
            self.env.close()


if __name__ == '__main__':
    path = sys.path[0].rsplit("/", 1)[0]
    algo = sac_algo(path)
    algo.train()