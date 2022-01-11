import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import sys
sys.path.append(".")
from args.config import ActorCritic_params as params


class ActorCritic(nn.Module):
    def __init__(self, learning_rate, gamma):
        super(ActorCritic, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.data = []

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + self.gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(
            self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


class ActorCritic_ALGO():
    def __init__(self):
        super(ActorCritic_ALGO, self).__init__()
        self.env = gym.make(params['gym_env'])
        self.print_interval = params["print_interval"]
        self.score = params["score"]
        self.epoch = params["epoch"]
        self.learning_rate = params["learning_rate"]
        self.gamma = params["gamma"]
        self.n_rollout = params["n_rollout"]
        self.model = ActorCritic(self.learning_rate, self.gamma)

    def init_write(self):
        with open("./result/ActorCritic.csv", "w+", encoding="utf-8") as f:
            f.write("epoch_number,reward\n")

    def train(self):

        for n_epi in range(10000):
            done = False
            s = self.env.reset()
            while not done:
                for t in range(self.n_rollout):
                    prob = self.model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, info = self.env.step(a)
                    self.model.put_data((s, a, r, s_prime, done))

                    s = s_prime
                    self.score += r

                    if done:
                        break

                self.model.train_net()

            with open("./result/ActorCritic.csv", "a+", encoding="utf-8") as f:
                f.write("{},{}\n".format(n_epi, self.score))

            if n_epi % self.print_interval == 0:
                print("episode :{}, avg_score : {:.1f}".format(
                    n_epi, self.score / self.print_interval))
                self.score = 0.0
        self.env.close()


if __name__ == '__main__':
    algo = ActorCritic_ALGO()
    algo.train()