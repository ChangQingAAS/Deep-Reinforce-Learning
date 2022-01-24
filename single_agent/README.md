## 实现基础的单智能体DRL算法

- 每个算法由一个文件编写.
- 尽量做到环境与参数一致

## 依赖

1. PyTorch 1.10
2. OpenAI GYM

## 算法

### 离散动作 - 》 “CartPole-v1”

1. REINFORCE
2. DQN(including replay memory and target network)
3. PPO (including GAE)
4. Actor Critic
5. DDQN
6. DuelingDQN

### 连续动作 -》 “Pendulum-v1"

1. DDPG(including OU noise and soft target update)
2. SAC

### TODO: 

1. acer
3. A3C
4. vtrace
5. A2C
6. PPO-Continuous
7. PPO-LSTM

## 运行方式

例如：

```bash
cd single_agent
python3 algo/DQN.py 
```
