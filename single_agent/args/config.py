REINFORCE_params = {
    "gym_env": 'CartPole-v1',
    "print_interval": 500,
    "learning_rate": 0.0002,
    "gamma": 0.98,
    "epoch": 10000,
    "score": 0.0,
}

ActorCritic_params = {
    "gym_env": 'CartPole-v1',
    "print_interval": 500,
    "learning_rate": 0.0002,
    "gamma": 0.98,
    "epoch": 10000,
    "score": 0.0,
    "n_rollout": 10,
}

dqn_params = {
    "gym_env": 'CartPole-v1',
    "print_interval": 500,
    "learning_rate": 0.0005,  # 这里不一样
    "gamma": 0.98,
    "epoch": 10000,
    "score": 0.0,
    "n_rollout": 10,
    "buffer_limit": 50000,
    "batch_size": 32
}
