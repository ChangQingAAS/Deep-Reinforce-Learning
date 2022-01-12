import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import sys
sys.path.append(".")
from args.config import a3c_params as params


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def train(global_model, rank, learning_rate, gamma, max_train_ep,
          update_interval):
    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

    env = gym.make(params['gym_env'])

    for n_epi in range(max_train_ep):
        done = False
        s = env.reset()
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                prob = local_model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r / 100.0)

                s = s_prime
                if done:
                    break

            s_final = torch.tensor(s_prime, dtype=torch.float)
            R = 0.0 if done else local_model.v(s_final).item()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                td_target_lst.append([R])
            td_target_lst.reverse()

            s_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                torch.tensor(td_target_lst)
            advantage = td_target - local_model.v(s_batch)

            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)
            loss = -torch.log(pi_a) * advantage.detach() + \
                F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())

            optimizer.zero_grad()
            loss.mean().backward()
            for global_param, local_param in zip(global_model.parameters(),
                                                 local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())

    env.close()
    print("Training process {} reached maximum episode.".format(rank))


def test(global_model, max_test_ep):
    env = gym.make(params['gym_env'])
    score = 0.0
    print_interval = 20

    for n_epi in range(max_test_ep):
        done = False
        s = env.reset()
        score = 0.0

        while not done:
            prob = global_model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().item()
            s_prime, r, done, info = env.step(a)
            s = s_prime
            score += r

        with open("./result/a3c.csv", "a+", encoding="utf-8") as f:
            f.write("{},{}\n".format(n_epi, score))

        time.sleep(1)
    env.close()


class a3c_algo():
    def __init__(self):
        super(a3c_algo, self).__init__()
        self.learning_rate = params['learning_rate']
        self.gamma = params['gamma']
        self.global_model = ActorCritic()
        self.global_model.share_memory()
        self.processes = []
        self.max_test_ep = params['max_test_ep']
        self.max_train_ep = params['max_train_ep']
        self.upadte_interval = params['update_interval']
        self.n_train_processes = params['n_train_processes']

    def init_write(self):
        with open("./result/a3c.csv", "w+", encoding="utf-8") as f:
            f.write("epoch_number,reward\n")

    def train(self):
        self.init_write()
        for rank in range(self.n_train_processes + 1):  # + 1 for test process
            if rank == 0:
                p = mp.Process(target=test,
                               args=(self.global_model, self.max_test_ep))
            else:
                p = mp.Process(target=train,
                               args=(self.global_model, rank,
                                     self.learning_rate, self.gamma,
                                     self.max_train_ep, self.upadte_interval))
            p.start()
            self.processes.append(p)
        for p in self.processes:
            p.join()


if __name__ == "__main__":
    algo = a3c_algo()
    algo.train()