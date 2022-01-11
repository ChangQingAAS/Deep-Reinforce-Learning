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

ppo_params = {
    "gym_env": 'CartPole-v1',
    "print_interval": 500,
    "learning_rate": 0.0005,  # 这里不一样
    "gamma": 0.98,
    "epoch": 10000,
    "score": 0.0,
    "n_rollout": 10,
    "buffer_limit": 50000,
    "batch_size": 32,
    "lmbda": 0.95,
    "eps_clip": 0.1,
    "K_epoch": 3,
    "T_horizon": 20
}

a2c_params = {
    "gym_env": 'CartPole-v1',
    "learning_rate": 0.0002,
    "gamma": 0.98,
    "epoch": 10000,
    "score": 0.0,
    "n_rollout": 10,
    "buffer_limit": 50000,
    "batch_size": 32,
    "lmbda": 0.95,
    "eps_clip": 0.1,
    "K_epoch": 3,
    "T_horizon": 20,
    "n_train_processes": 3,
    "update_interval": 5,
    "max_train_steps": 60000,
    "print_interval": 500,
}

a3c_params = {
    "gym_env": 'CartPole-v1',
    "learning_rate": 0.0002,
    "gamma": 0.98,
    "epoch": 10000,
    "score": 0.0,
    "n_rollout": 10,
    "buffer_limit": 50000,
    "batch_size": 32,
    "lmbda": 0.95,
    "eps_clip": 0.1,
    "K_epoch": 3,
    "T_horizon": 20,
    "n_train_processes": 3,
    "update_interval": 5,
    "max_train_steps": 60000,
    "print_interval": 500,
    "max_train_ep": 300,
    "max_test_ep": 400
}

acer_params = {
    "gym_env": 'CartPole-v1',
    "learning_rate": 0.0002,
    "gamma": 0.98,
    "epoch": 10000,
    "score": 0.0,
    "n_rollout": 10,
    "buffer_limit": 6000,  ###
    "rollout_len": 10,
    "batch_size":
    4,  # Indicates 4 sequences per mini-batch (4*rollout_len = 40 samples total)
    "c": 1.0,  # For truncating importance sampling ratio
    "lmbda": 0.95,
    "eps_clip": 0.1,
    "K_epoch": 3,
    "T_horizon": 20,
    "n_train_processes": 3,
    "update_interval": 5,
    "max_train_steps": 60000,
    "print_interval": 20,  ###
    "max_train_ep": 300,
    "max_test_ep": 400,
}

ddpg_params = {
    "gym_env": 'Pendulum-v0',  ####
    "learning_rate": 0.0002,
    "gamma": 0.99,  ###
    "epoch": 10000,
    "score": 0.0,
    "n_rollout": 10,
    "buffer_limit": 50000,
    "batch_size": 32,
    "lmbda": 0.95,
    "eps_clip": 0.1,
    "K_epoch": 3,
    "T_horizon": 20,
    "n_train_processes": 3,
    "update_interval": 5,
    "max_train_steps": 60000,
    "print_interval": 20,  ###
    "max_train_ep": 300,
    "max_test_ep": 400,
    "lr_mu": 0.0005,
    "lr_q": 0.001,
    "tau": 0.005,  # for target network soft update
}

ppo_continuous_params = {
    "gym_env": 'Pendulum-v0',  ###
    "print_interval": 20,  ##
    "learning_rate": 0.0003,  ##
    "gamma": 0.9,  ##
    "epoch": 10000,
    "lmbda": 0.9,
    "eps_clip": 0.2,
    "K_epoch": 10,
    "rollout_len": 3,
    "buffer_size": 30,
    "minibatch_size": 32,
}

ppo_lstm_params = {
    "gym_env": 'CartPole-v1',
    "print_interval": 20,  ##
    "learning_rate": 0.0005,  ##
    "gamma": 0.98,
    "epoch": 10000,
    "lmbda": 0.95,  ##
    "eps_clip": 0.1,  ##
    "K_epoch": 2,  ##
    "T_horizon": 20,
    "print_epoch": 20,
}

sac_params = {
    "gym_env": 'Pendulum-v0',
    "print_interval": 20,  ##
    "gamma": 0.98,
    "epoch": 10000,
    "batch_size": 32,
    "buffer_limit": 50000,
    "tau": 0.01,  # for target network soft update
    "lr_pi": 0.0005,
    "lr_q": 0.001,
    "init_alpha": 0.01,
    "buffer_limit": 50000,
    "target_entropy": -1.0,  # for automated alpha update
    "lr_alpha": 0.001  # for automated alpha update
}