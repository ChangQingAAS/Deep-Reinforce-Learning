import gym
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import sys

sys.path.append(".")
from args.config import reinforce_params as params


class Policy(nn.Module):

    def __init__(self, learning_rate, gamma, in_dim, out_dim):
        super(Policy, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.data = []

        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)
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

    def __init__(self, path):
        super(REINFORCE_ALGO, self).__init__()
        self.path = path
        self.env = gym.make(params['gym_env'])
        self.print_interval = params["print_interval"]
        self.epoch = params["epoch"]
        self.learning_rate = params["learning_rate"]
        self.gamma = params["gamma"]
        self.train_number = params['train_number']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.pi = Policy(self.learning_rate, self.gamma, self.obs_dim, self.action_dim)

        self.init_write()

    def init_write(self):
        for i in range(self.train_number):
            with open(self.path + "/result/REINFORCE/result_%s.csv" % str(i), "w+", encoding="utf-8") as f:
                f.write("epoch_number,average reward\n")

    def train(self):
        for train_counter in range(self.train_number):
            score = 0.0
            for self.n_epi in range(self.epoch):
                self.s = self.env.reset()
                self.done = False

                while not self.done:  # CartPole-v1 forced to terminates at 500 step.
                    self.prob = self.pi(torch.from_numpy(self.s).float())
                    self.m = Categorical(self.prob)
                    self.a = self.m.sample()
                    self.s_prime, self.r, self.done, self.info = self.env.step(self.a.item())
                    self.pi.put_data((self.r, self.prob[self.a]))
                    self.s = self.s_prime
                    score += self.r

                self.pi.train_net()

                if self.n_epi % self.print_interval == 0:
                    with open(self.path + "/result/REINFORCE/result_%s.csv" % str(train_counter),
                              "a+",
                              encoding="utf-8") as f:
                        f.write("{},{}\n".format(self.n_epi, score / self.print_interval))
                        print("{},{}\n".format(self.n_epi, score / self.print_interval))
                    score = 0.0
            self.env.close()


if __name__ == "__main__":
    path = sys.path[0].rsplit("/", 1)[0]
    algo = REINFORCE_ALGO(path)
    algo.train()
