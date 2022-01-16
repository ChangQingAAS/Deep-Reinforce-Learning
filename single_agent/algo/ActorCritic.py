import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import sys

sys.path.append(".")
from args.config import ac_params as params


class Actor(nn.Module):

    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Critic(nn.Module):

    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorCritic_ALGO():

    def __init__(self, path):
        self.path = path
        self.env = gym.make(params['gym_env'])
        self.gamma = params['gamma']
        self.learning_rate = params['learning_rate']
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.actor = Actor(self.obs_dim, self.act_dim)
        self.critic = Critic(self.obs_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), self.learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), self.learning_rate)
        self.criterion = nn.MSELoss()
        self.epoch = params['epoch']
        self.batch_size = params['batch_size']
        self.print_interval = params['print_interval']
        self.train_number = params['train_number']

        self.init_write()

    def init_write(self):
        for i in range(self.train_number):
            with open(self.path + "/result/AC/result_%s.csv" % str(i), "w+",
                      encoding="utf-8") as f:
                f.write("epoch_number,average reward\n")

    def train(self):
        for train_counter in range(self.train_number):
            cumulative_reward = 0.0
            rewards = []
            states = []
            next_states = []
            dones = []
            actions = []

            state = self.env.reset()

            for episode_count in range(self.epoch):
                state_t = torch.FloatTensor([state])
                with torch.no_grad():
                    probs = self.actor(state_t)
                    dist = Categorical(probs)
                    action = dist.sample().item()

                next_state, reward, done, _ = self.env.step(action)
                cumulative_reward += reward

                rewards.append([reward])
                states.append(state)
                actions.append([action])
                next_states.append(next_state)
                dones.append(done)

                state = next_state

                if len(rewards) == self.batch_size:
                    rewards_t = torch.FloatTensor(rewards)
                    actions_t = torch.LongTensor(actions)
                    with torch.no_grad():
                        next_states_t = torch.FloatTensor(next_states)
                        states_t = torch.FloatTensor(states)
                        next_states_v = self.critic(next_states_t)
                        states_v = self.critic(states_t)
                    next_states_v[dones] = 0.0
                    td_errors = self.gamma * next_states_v + rewards_t - states_v

                    self.optimizer_actor.zero_grad()
                    log_probs = torch.log(torch.gather(self.actor(states_t), 1, actions_t))
                    gradients = -td_errors * log_probs
                    gradient = gradients.mean()
                    gradient.backward()
                    self.optimizer_actor.step()

                    self.optimizer_critic.zero_grad()
                    target = self.gamma * next_states_v + rewards_t
                    outputs = self.critic(states_t)
                    loss = self.criterion(outputs, target)
                    loss.backward()
                    self.optimizer_critic.step()

                    rewards = []
                    states = []
                    next_states = []
                    dones = []
                    actions = []

                if done and episode_count % self.print_interval == 0:
                    state = self.env.reset()
                    state_t = torch.FloatTensor([state])
                    with open(self.path + "/result/AC/result_%s.csv" % str(train_counter),
                              "a+",
                              encoding="utf-8") as f:
                        f.write("{},{}\n".format(episode_count,
                                                 cumulative_reward / self.print_interval))

                    print('episode %d, reward: %.3f' %
                          (episode_count, cumulative_reward / self.print_interval))
                    cumulative_reward = 0

                self.env.close()


if __name__ == '__main__':
    path = sys.path[0].rsplit("/", 1)[0]
    algo = ActorCritic_ALGO(path)
    algo.train()