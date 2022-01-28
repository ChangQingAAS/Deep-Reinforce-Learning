## 实现基础的单智能体DRL算法

- 每个算法由一个文件编写.
- 尽量做到环境与参数一致

## 依赖

- Ubuntu 20.04
- python 3.8.10
- PyTorch 1.10.1
- OpenAI GYM 0.21.1  

## 文件说明
- algo文件夹: 
    - 具体算法
- args文件夹: 
    - 各个算法的参数（没用argparse主要是因为重构起来太麻烦
- result 文件夹：
    - 保存算法运行后的结果
- vis 文件夹：
    - 算法结果可视化

## 算法

### 离散动作 - 》 “CartPole-v1”

1. REINFORCE
2. DQN(including replay memory and target network)
3. PPO (including GAE)
4. Actor Critic
5. DDQN
6. DuelingDQN
7. PPO-LSTM

### 连续动作 -》 “Pendulum-v1"

1. DDPG(including OU noise and soft target update)
2. SAC

### TODO: 

1. acer
3. A3C
4. vtrace
5. A2C
6. PPO-Continuous

## 运行方式

例如：

```bash
cd single_agent
python3 algo/DQN.py 
```

## others
- 为方便对比所有算法的参数，请在args/config.py里定义参数()
- 上面标TODO的算法也实现了，只是效果一般，就暂时没有进行一些工程上的重构优化，落后其他算法一个版本
