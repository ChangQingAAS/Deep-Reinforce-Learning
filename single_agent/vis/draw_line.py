import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import sys
import os


def get_all_data(algo_list, path):
    x = {}
    y = {}
    for algo in algo_list:
        x[algo] = []
        y[algo] = []
        for items in os.walk(path + '/result/%s' % algo):
            files = sorted(items[2])  #当前路径下所有非目录子文件
        for item in files:
            with open(path + '/result/%s/' % algo + item, "r+", encoding="utf-8") as csvfile:
                epoch_number = []
                average_reward = []
                reader = csv.DictReader(csvfile)
                for row in reader:
                    average_reward.append(float(row['average reward']))
                    if x[algo] == []:
                        epoch_number.append(int(row['epoch_number']))
            y[algo].append(average_reward)
            if x[algo] == []:
                x[algo].append(epoch_number)
    return x, y


def draw_single(algo_list, color, x, y, env, path):
    for i, algo in enumerate(algo_list):
        plt.cla()
        fig = plt.figure(dpi=600)
        sns.tsplot(time=x[algo], data=y[algo], condition=algo, linewidth=0.5, color=color[i])
        plt.ylabel('average reward', fontsize=10)
        plt.xlabel('num episodes', fontsize=10)
        plt.title(env, fontsize=10)
        plt.legend(loc='lower right', fontsize=5)
        plt.savefig(path + "/vis/%s--%s.png" % (algo, env))


def draw_all(algo_list, color, x, y, env, path):
    plt.cla()
    fig = plt.figure(dpi=600)
    for i, algo in enumerate(reversed(algo_list)):
        sns.tsplot(time=x[algo], data=y[algo], condition=algo, linewidth=0.5, color=color[i])
        plt.ylabel('average reward', fontsize=10)
        plt.xlabel('num episodes', fontsize=10)
        # plt.yticks((0, 500, 100))
        # plt.xticks((0, 2000, 20))
        plt.title(env, fontsize=10)
        plt.legend(loc='lower right', fontsize=5)
        plt.savefig(path + "/vis/ALL-%s.png" % (env))


if __name__ == "__main__":
    path = sys.path[0].rsplit("/", 1)[0]
    env = "Pendulum-v1"
    if env == 'CartPole-v1':
        # algo_list = ["vtrace",  "acer", "a3c", "PPO_lstm"]
        algo_list = ['REINFORCE', 'DQN', 'PPO', 'DDQN', 'AC', 'DuelingDQN', 'PPO-LSTM']
    elif env == 'Pendulum-v1':
        algo_list = ['DDPG', 'SAC']
    x, y = get_all_data(algo_list, path)
    color = sns.hls_palette(len(algo_list), l=.5, s=.5)  # l-亮度 lightness s-饱和 saturation
    draw_single(algo_list, color, x, y, env, path)
    draw_all(algo_list, color, x, y, env, path)
