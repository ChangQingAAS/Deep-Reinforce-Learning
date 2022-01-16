import seaborn as sns
import matplotlib.pyplot as plt
import csv
import sys
import os


def get_data(path):
    x = []
    y = []
    for items in os.walk(path + '/result'):
        files = sorted(items[2])  # 当前路径下所有非目录子文件
    for item in files:
        with open(path + '/result/' + item, "r+", encoding="utf-8") as csvfile:
            epoch_number = []
            average_reward = []
            reader = csv.DictReader(csvfile)
            for row in reader:
                average_reward.append(float(row['average reward']))
                if x == []:
                    epoch_number.append(int(row['epoch_number']))
        y.append(average_reward)
        if x == []:
            x.append(epoch_number)

    return x, y


def draw_single(algo_name, color, x, y, env, path):
    plt.cla()
    sns.tsplot(time=x, data=y, condition=algo_name, linewidth=0.5, color=color)
    plt.ylabel('average reward', fontsize=10)
    plt.xlabel('num episodes', fontsize=10)
    plt.title(env, fontsize=10)
    plt.legend(loc='lower right', fontsize=5)
    plt.savefig(path + "/vis/%s--%s.png" % (algo_name, env))


if __name__ == "__main__":
    path = sys.path[0].rsplit("/", 1)[0]
    x, y = get_data(path)
    scenario = 'simple_adversary'
    draw_single('MADDPG', 'b', x, y, scenario, path)
