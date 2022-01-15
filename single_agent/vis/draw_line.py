import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import random


def get_all_data(algo_list):
    x = {}
    y = {}
    for algo in algo_list:
        x[algo] = []
        y[algo] = []
        with open("result/%s.csv" % algo, "r+", encoding="utf-8") as csvfile:
            epoch_number = []
            average_reward = []
            reader = csv.DictReader(csvfile)
            for row in reader:
                epoch_number.append(int(row['epoch_number']))
                average_reward.append(float(row['average reward']))
        x[algo].append(epoch_number)
        y[algo].append(average_reward)
    return x, y


def draw_single(algo_list, color, x, y, env):
    for i, algo in enumerate(algo_list):
        plt.cla()
        sns.tsplot(time=x[algo], data=y[algo], condition=algo, linewidth=0.5, color=color[i])
        plt.ylabel('average reward', fontsize=10)
        plt.xlabel('num episodes', fontsize=10)
        plt.title(env, fontsize=10)
        plt.legend(loc='lower right', fontsize=5)
        plt.savefig("vis/%s--%s.png" % (algo, env))


def draw_all(algo_list, color, x, y, env):
    plt.cla()
    for i, algo in enumerate(reversed(algo_list)):
        sns.tsplot(time=x[algo], data=y[algo], condition=algo, linewidth=0.5, color=color[i])
        plt.ylabel('average reward', fontsize=10)
        plt.xlabel('num episodes', fontsize=10)
        # plt.yticks((0, 500, 100))
        # plt.xticks((0, 2000, 20))
        plt.title(env, fontsize=10)
        plt.legend(loc='lower right', fontsize=5)
        plt.savefig("vis/ALL-%s.png" % (env))


if __name__ == "__main__":
    env = "CartPole-v1"
    # algo_list = ['PPO', 'REINFORCE', 'DQN', "DDPG", "vtrace", 'ActorCritic', "acer", "a3c", "PPO_lstm"]
    algo_list = ['DQN', 'PPO']
    x, y = get_all_data(algo_list)
    color = sns.hls_palette(len(algo_list), l=.5, s=.5)  # l-亮度 lightness s-饱和 saturation
    draw_single(algo_list, color, x, y, env)
    draw_all(algo_list, color, x, y, env)
