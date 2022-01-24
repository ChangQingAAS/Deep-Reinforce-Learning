from ActorCritic import ActorCritic_ALGO
from DDPG import DDPG_algo
from DDQN import DDQN_ALGO
from DQN import DQN_ALGO
from Dueling_dqn import DuelingDQN_ALGO
from PPO import PPO_algo
from REINFORCE import REINFORCE_ALGO
from sac import sac_algo
import sys

path = sys.path[0].rsplit("/", 1)[0]


algo = DQN_ALGO(path)
algo.train()

algo = ActorCritic_ALGO(path)
algo.train()

algo = DDPG_algo(path)
algo.train()

algo = REINFORCE_ALGO(path)
algo.train()

algo = DuelingDQN_ALGO(path)
algo.train()

algo = DDQN_ALGO(path)
algo.train()

algo = PPO_algo(path)
algo.train()

algo = sac_algo(path)
algo.train()