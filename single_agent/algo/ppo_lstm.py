#PPO-LSTM
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np
import sys
sys.path.append(".")
from args.config import PPO_lstm_params as params


class PPO(nn.Module):
    def __init__(self, learning_rate, K_epoch, gamma, lmbda, eps_clip):
        super(PPO, self).__init__()
        self.learning_rate = learning_rate
        self.K_epoch = K_epoch
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.data = []

        self.fc1 = nn.Linear(4, 64)
        self.lstm = nn.LSTM(64, 32)
        self.fc_pi = nn.Linear(32, 2)
        self.fc_v = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden

    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (
            h1_out, h2_out) = self.make_batch()
        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(self.K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + self.gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1, a)
            ratio = torch.exp(torch.log(pi_a) -
                              torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(
                v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()


class PPO_lstm_algo():
    def __init__(self):
        super(PPO_lstm_algo, self).__init__()
        self.env = gym.make(params['gym_env'])
        self.T_horizon = params['T_horizon']
        self.epoch = params['epoch']
        self.learning_rate = params['learning_rate']
        self.K_epoch = params['K_epoch']
        self.gamma = params['gamma']
        self.eps_clip = params['eps_clip']
        self.lmbda = params['lmbda']
        self.print_interval = params["print_interval"]
        self.model = PPO(self.learning_rate, self.K_epoch, self.gamma,
                         self.lmbda, self.eps_clip)

        self.print_interval = params['print_interval']
        self.init_write()

    def init_write(self):
        with open("./result/PPO_lstm.csv", "w+", encoding="utf-8") as f:
            f.write("epoch_number,average reward\n")

    def train(self):
        score = 0.0
        for n_epi in range(self.epoch):
            h_out = (torch.zeros([1, 1, 32], dtype=torch.float),
                     torch.zeros([1, 1, 32], dtype=torch.float))
            s = self.env.reset()
            done = False

            while not done:
                for t in range(self.T_horizon):
                    h_in = h_out
                    prob, h_out = self.model.pi(
                        torch.from_numpy(s).float(), h_in)
                    prob = prob.view(-1)
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, info = self.env.step(a)

                    self.model.put_data(
                        (s, a, r, s_prime, prob[a].item(), h_in, h_out, done))
                    s = s_prime

                    score += r
                    if done:
                        break

                self.model.train_net()

            if n_epi % self.print_interval == 0:
                with open("./result/PPO_lstm.csv", "a+",
                          encoding="utf-8") as f:
                    f.write("{},{:.1f} \n".format(n_epi,
                                                  score / self.print_interval))
                print("episode :{}, avg score : {:.1f}".format(
                    n_epi, score / self.print_interval))
                score = 0.0

        self.env.close()


if __name__ == '__main__':
    algo = PPO_lstm_algo()
    algo.train()