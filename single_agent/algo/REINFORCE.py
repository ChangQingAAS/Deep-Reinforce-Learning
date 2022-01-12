import gym
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import sys
sys.path.append(".")
from args.config import default_params as params


class Policy(nn.Module):
    def __init__(self, learning_rate, gamma):
        super(Policy, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


class REINFORCE_ALGO():
    def __init__(self):
        super(REINFORCE_ALGO, self).__init__()
        self.env = gym.make(params['gym_env'])
        self.print_interval = params["print_interval"]
        self.epoch = params["epoch"]
        self.learning_rate = params["learning_rate"]
        self.gamma = params["gamma"]
        self.pi = Policy(self.learning_rate, self.gamma)

    def init_write(self):
        with open("./result/REINFORCE.csv", "w+", encoding="utf-8") as f:
            f.write("epoch_number,average reward\n")

    def train(self):
        self.init_write()
        score = 0.0
        for self.n_epi in range(self.epoch):
            self.s = self.env.reset()
            self.done = False

            while not self.done:  # CartPole-v1 forced to terminates at 500 step.
                self.prob = self.pi(torch.from_numpy(self.s).float())
                self.m = Categorical(self.prob)
                self.a = self.m.sample()
                self.s_prime, self.r, self.done, self.info = self.env.step(
                    self.a.item())
                self.pi.put_data((self.r, self.prob[self.a]))
                self.s = self.s_prime
                score += self.r

            self.pi.train_net()

            if self.n_epi % self.print_interval == 0:
                with open("./result/REINFORCE.csv", "a+",
                          encoding="utf-8") as f:
                    f.write("{},{}\n".format(self.n_epi,
                                             score / self.print_interval))
                    print("{},{}\n".format(self.n_epi,
                                           score / self.print_interval))
                score = 0.0

        self.env.close()


if __name__ == "__main__":
    algo = REINFORCE_ALGO()
    algo.train()
