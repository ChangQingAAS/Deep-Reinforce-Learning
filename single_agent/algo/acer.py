import gym
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import sys
sys.path.append(".")
from args.config import acer_params as params

# Characteristics
# 1. Discrete action space, single thread version.
# 2. Does not suPPOrt trust-region updates.


class ReplayBuffer():
    def __init__(self, buffer_limit, batch_size):
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, seq_data):
        self.buffer.append(seq_data)

    def sample(self, on_policy=False):
        if on_policy:
            mini_batch = [self.buffer[-1]]
        else:
            mini_batch = random.sample(self.buffer, self.batch_size)

        s_lst, a_lst, r_lst, prob_lst, done_lst, is_first_lst = [], [], [], [], [], []
        for seq in mini_batch:
            is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            for transition in seq:
                s, a, r, prob, done = transition

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r)
                prob_lst.append(prob)
                done_mask = 0.0 if done else 1.0
                done_lst.append(done_mask)
                is_first_lst.append(is_first)
                is_first = False

        s,a,r,prob,done_mask,is_first = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                        r_lst, torch.tensor(prob_lst, dtype=torch.float), done_lst, \
                                        is_first_lst
        return s, a, r, prob, done_mask, is_first

    def size(self):
        return len(self.buffer)


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_q = nn.Linear(256, 2)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        pi = F.softmax(x, dim=softmax_dim)
        return pi

    def q(self, x):
        x = F.relu(self.fc1(x))
        q = self.fc_q(x)
        return q


def train(
    model,
    optimizer,
    memory,
    gamma,
    c,
    on_policy=False,
):
    s, a, r, prob, done_mask, is_first = memory.sample(on_policy)

    q = model.q(s)
    q_a = q.gather(1, a)
    pi = model.pi(s, softmax_dim=1)
    pi_a = pi.gather(1, a)
    v = (q * pi).sum(1).unsqueeze(1).detach()

    rho = pi.detach() / prob
    rho_a = rho.gather(1, a)
    rho_bar = rho_a.clamp(max=c)
    correction_coeff = (1 - c / rho).clamp(min=0)

    q_ret = v[-1] * done_mask[-1]
    q_ret_lst = []
    for i in reversed(range(len(r))):
        q_ret = r[i] + gamma * q_ret
        q_ret_lst.append(q_ret.item())
        q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]

        if is_first[i] and i != 0:
            q_ret = v[i - 1] * done_mask[
                i - 1]  # When a new sequence begins, q_ret is initialized

    q_ret_lst.reverse()
    q_ret = torch.tensor(q_ret_lst, dtype=torch.float).unsqueeze(1)

    loss1 = -rho_bar * torch.log(pi_a) * (q_ret - v)
    loss2 = -correction_coeff * pi.detach() * torch.log(pi) * (
        q.detach() - v)  # bias correction term
    loss = loss1 + loss2.sum(1) + F.smooth_l1_loss(q_a, q_ret)

    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()


class acer_algo():
    def __init__(self):
        super(acer_algo, self).__init__()
        self.env = gym.make(params['gym_env'])
        self.learning_rate = params['learning_rate']
        self.buffer_limit = params['buffer_limit']
        self.batch_size = params['batch_size']
        self.memory = ReplayBuffer(self.buffer_limit, self.batch_size)
        self.model = ActorCritic()
        self.print_interval = params['print_interval']
        self.rollout_len = params['rollout_len']
        self.c = params['c']
        self.gamma = params['gamma']
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

    def init_write(self):
        with open("./result/acer.csv", "w+", encoding="utf-8") as f:
            f.write("epoch_number,average reward\n")

    def train(self):
        self.init_write()
        for n_epi in range(10000):
            score = 0.0
            s = self.env.reset()
            done = False
            while not done:
                seq_data = []
                for t in range(self.rollout_len):
                    prob = self.model.pi(torch.from_numpy(s).float())
                    a = Categorical(prob).sample().item()
                    s_prime, r, done, info = self.env.step(a)
                    seq_data.append((s, a, r, prob.detach().numpy(), done))

                    score += r
                    s = s_prime
                    if done:
                        break

                self.memory.put(seq_data)
                if self.memory.size() > 500:
                    train(self.model,
                          self.optimizer,
                          self.memory,
                          self.gamma,
                          self.c,
                          on_policy=True)
                    train(self.model, self.optimizer, self.memory, self.gamma,
                          self.c)

            if n_epi % self.print_interval == 0:
                with open("./result/acer.csv", "a+", encoding="utf-8") as f:
                    f.write("{},{}\n".format(n_epi,
                                             score / self.print_interval))
                print("n_episode :{}, score : {:.1f}".format(
                    n_epi, score / self.print_interval))
                score = 0.0
        self.env.close()


if __name__ == '__main__':
    algo = acer_algo()
    algo.train()