import seaborn as sns
import matplotlib.pyplot as plt
import csv
import sys
import pandas as pd
import numpy as np


def xlsx_to_csv_pd(algo, path):
    data_xls = pd.read_excel(path + "/result/%s.xlsx" % algo, index_col=0)
    data_xls.to_csv(path + "/result/%s.csv" % algo, encoding='utf-8')


def get_all_data(algo_list, path):
    x = {}
    y = {}
    for algo in algo_list:
        xlsx_to_csv_pd(algo, path)
        y[algo] = []
        x[algo] = np.loadtxt(open(path + "/result/%s.csv" % algo, "r"), delimiter=",", skiprows=1, usecols=[0])
        with open(path + "/result/%s.csv" % algo, "r+", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                test_number = len(row) - 1
                break

        for test_counter in range(1, test_number):
            y[algo].append(
                np.loadtxt(open(path + "/result/%s.csv" % algo, "r"),
                           delimiter=",",
                           skiprows=1,
                           usecols=[test_counter]))
    return x, y


def draw_single(algo_list, color, x, y, env, path):
    for i, algo in enumerate(algo_list):
        plt.cla()
        sns.tsplot(time=x[algo], data=y[algo], condition=algo, linewidth=0.5, color=color[i])
        plt.ylabel('average reward', fontsize=10)
        plt.xlabel('num episodes', fontsize=10)
        plt.title(env, fontsize=10)
        plt.legend(loc='lower right', fontsize=5)
        plt.savefig(path + "/vis/%s--%s.png" % (algo, env))


if __name__ == "__main__":
    path = sys.path[0].rsplit("/", 1)[0]
    env = "petting_zoo"
    algo_list = ['maddpg']
    x, y = get_all_data(algo_list, path)
    color = sns.hls_palette(len(algo_list), l=.3, s=.8)
    draw_single(algo_list, color, x, y, env, path)
