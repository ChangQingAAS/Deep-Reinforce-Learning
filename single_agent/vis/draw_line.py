from typing import Counter
import matplotlib.pyplot as plt
import csv


def draw_all(env, algo_name_list):
    for algo_name in algo_name_list:
        x = []
        y = []
        with open("./result/%s.csv" % algo_name, "r+", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x.append(int(row['epoch_number']))
                y.append(float(row['average reward']))
            plt.plot(x, y, label=algo_name, linewidth=0.5)

    plt.ylabel('average reward')
    plt.xlabel('num episodes')
    # plt.yticks((0, 500, 100))
    # plt.xticks((0, 2000, 20))
    plt.title(env)
    plt.legend(loc='lower right', fontsize=5)  # 标签位置
    plt.savefig("./vis/all-%s.png" % env)


def draw_single(env, algo_name):
    plt.cla()
    x = []
    y = []
    with open("./result/%s.csv" % algo_name, "r+", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x.append(int(row['epoch_number']))
            y.append(float(row['average reward']))
        plt.plot(x, y, label=algo_name, linewidth=0.5)

    plt.ylabel('average reward')
    plt.xlabel('num episodes')
    # plt.yticks((0, 500, 100))
    # plt.xticks((0, 2000, 20))
    plt.title(env)
    plt.legend(loc='lower right', fontsize=5)  # 标签位置
    plt.savefig("./vis/%s-%s.png" % (algo_name, env))


if __name__ == "__main__":
    env = 'CartPole-v1'
    algo_name_list = ['PPO', 'REINFORCE', 'DQN', "DDPG", "vtrace", 'ActorCritic', "acer", "a3c", "ppo_lstm"]

    for item in algo_name_list:
        draw_single(env, item)

    draw_all(env, algo_name_list)
