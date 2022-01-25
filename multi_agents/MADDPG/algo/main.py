import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v2, simple_v2
from config import *
import sys


class MADDPG_ALGO():

    def __init__(self, path):
        self.scenario = scenario
        self.path = path
        self.total_steps = total_steps
        self.fc1 = fc1
        self.fc2 = fc2
        self.alpha = alpha
        self.beta = beta
        self.chkpt_dir = chkpt_dir
        self.buffer_limit = buffer_limit
        self.batch_size = batch_size
        self.evaluate = evaluate
        self.epoch = epoch
        self.train_number = train_number
        if self.scenario == "simple_adversary":
            self.env = simple_adversary_v2.parallel_env()
            self.env.reset()
        else:
            self.env = simple_adversary_v2.parallel_env()
            self.env.reset()

        self.critic_obs_dims = 0
        for i in self.env.agents:
            self.critic_obs_dims += self.env.observation_spaces[i].shape[0]

        self.n_actions = []
        for i in self.env.agents:
            self.n_actions.append(self.env.action_spaces[i].n)

        self.actor_obs_dims = []
        for i in self.env.agents:
            self.actor_obs_dims.append(self.env.observation_spaces[i].shape[0])

        self.maddpg_agents = MADDPG(self.env, self.fc1, self.fc2, self.alpha, self.beta, self.chkpt_dir)
        self.memory = MultiAgentReplayBuffer(self.env, max_size=self.buffer_limit, batch_size=self.batch_size)

        if evaluate:
            self.maddpg_agents.load_checkpoint()

        self.init_write()

    def init_write(self):
        for i in range(self.train_number):
            with open(self.path + "/result/result_%s.csv" % str(i), "w+", encoding="utf-8") as f:
                f.write("epoch_number,average reward\n")

    def obs_list_to_state_vector(self, observation):
        state = np.array([])
        for agent in self.env.agents:
            state = np.concatenate([state, observation[agent]])
        return state

    def train(self):
        for train_counter in range(train_number):
            for epoch_number in range(self.epoch):
                obs = self.env.reset()
                # done =self.env.dones
                done_list = [False for agent in self.env.agents]

                score = 0
                episode_step = 0

                while not any(done_list):
                    # env.render(mode='human')
                    actions = self.maddpg_agents.choose_action(obs, False)
                    # print("the actions:", actions)
                    obs_, reward, done, infos = self.env.step(actions)
                    done_list = list(done.values())
                    if episode_step > MAX_STEPS:
                        done_list = [True]

                    state = self.obs_list_to_state_vector(obs)
                    state_ = self.obs_list_to_state_vector(obs_)

                    self.memory.store_transition(obs, state, actions, reward, obs_, state_, done)

                    obs = obs_

                    episode_step += 1
                    self.total_steps += 1
                    score += sum(reward.values())  #  I have a feeling it collects the rewards

                    # got to try and get it to learn now
                    if self.total_steps % 100 == 0:
                        self.maddpg_agents.learn(self.memory)

                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
                # if not evaluate:
                #    if avg_score > best_score:
                #       maddpg_agents.save_checkpoint()
                #      best_score = avg_score
                if epoch_number % PRINT_INTERVAL == 0:
                    print('episode', epoch_number, 'average score {:.1f}'.format(avg_score))
                    with open(self.path + "/result/result_%s.csv" % str(train_counter), "a+", encoding="utf-8") as f:
                        f.write("{},{}\n".format(epoch_number, avg_score))

        self.env.close()


if __name__ == "__main__":
    path = sys.path[0].rsplit("/", 1)[0]
    algo = MADDPG_ALGO(path)
    algo.train()
