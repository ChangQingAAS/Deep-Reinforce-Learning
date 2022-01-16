alpha = 0.01
beta = 0.01
fc1 = 64
fc2 = 64
gamma = 0.95
tau = 0.01

PRINT_INTERVAL = 20
epoch = 1000 
MAX_STEPS = 50
total_steps = 0
score_history = []
evaluate = False
best_score = 0

train_number = 3
scenario = 'simple_adversary'

chkpt_dir = 'testing_save/'

batch_size = 1024  # 重放缓冲区batch
buffer_limit = 100000  # 重放缓冲区大小
