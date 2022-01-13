import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import sys
sys.path.append(".")
from args.config import ppo_continuous_params as params


class PPO(nn.Module):
    def __init__(self, learning_rate, minibatch_size, gamma, lmbda, eps_clip,
                 K_epoch, buffer_size):
        super(PPO, self).__init__()
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.buffer_size = buffer_size
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.data = []

        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.fc_v = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(self.buffer_size):
            for i in range(self.minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition

                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)

            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                          torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                          torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob,
                                  td_target, advantage))

        return data_with_adv

    def train_net(self):
        if len(self.data) == self.minibatch_size * self.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(self.K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(
                        log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip,
                                        1 + self.eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(
                        self.v(s), td_target)

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1


class ppo_continuous_algo():
    def __init__(self):
        super(ppo_continuous_algo, self).__init__()
        self.env = gym.make(params['gym_env'])
        self.rollout_len = params['rollout_len']
        self.epoch = params['epoch']
        self.print_interval = params['print_interval']
        self.learning_rate = params['learning_rate']
        self.minibatch_size = params['minibatch_size']
        self.gamma = params['gamma']
        self.lmbda = params['lmbda']
        self.eps_clip = params['eps_clip']
        self.K_epoch = params['K_epoch']
        self.buffer_size = params['buffer_size']

        self.model = PPO(self.learning_rate, self.minibatch_size, self.gamma,
                         self.lmbda, self.eps_clip, self.K_epoch,
                         self.buffer_size)

        self.rollout = []

        self.init_write()

    def init_write(self):
        with open("./result/ppo_continuous.csv", "w+", encoding="utf-8") as f:
            f.write("epoch_number,average reward,optimization_steps\n")

    def train(self):
        for n_epi in range(self.epoch):
            score = 0.0
            s = self.env.reset()
            done = False
            while not done:
                for t in range(self.rollout_len):
                    mu, std = self.model.pi(torch.from_numpy(s).float())
                    dist = Normal(mu, std)
                    a = dist.sample()
                    log_prob = dist.log_prob(a)
                    s_prime, r, done, info = self.env.step([a.item()])

                    self.rollout.append(
                        (s, a, r / 10.0, s_prime, log_prob.item(), done))
                    if len(self.rollout) == self.rollout_len:
                        self.model.put_data(self.rollout)
                        self.rollout = []

                    s = s_prime
                    score += r
                    if done:
                        break

                self.model.train_net()

            if n_epi % self.print_interval == 0:
                print("episode :{}, avg score : {:.1f}, opt step: {}".
                      format(n_epi, score / self.print_interval,
                             self.model.optimization_step))
                with open("./result/ppo_continuous.csv",
                          "a+",
                          encoding="utf-8") as f:
                    f.write("{},{:.1f},{}\n".format(
                        n_epi, score / self.print_interval,
                        self.model.optimization_step))
            score = 0.0

        self.env.close()


if __name__ == '__main__':
    algo = ppo_continuous_algo()
    algo.train()