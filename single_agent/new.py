import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def getdata():
    basecond = [[18, 20, 19, 18, 13, 4, 1], [20, 17, 12, 9, 3, 0, 0], [20, 20, 20, 12, 5, 3, 0]]

    cond1 = [[18, 19, 18, 19, 20, 15, 14], [19, 20, 18, 16, 20, 15, 9], [19, 20, 20, 20, 17, 10, 0],
             [20, 20, 20, 20, 7, 9, 1]]
    return basecond, cond1


data = getdata()
fig = plt.figure()
xdata = np.array([0, 1, 2, 3, 4, 5, 6]) / 5
linestyle = [
    '-',
    '--',
]
color = [
    'r',
    'g',
]
label = ['algo1', 'algo2']

for i in range(2):
    sns.tsplot(time=xdata, data=data[i], color=color[i], linestyle=linestyle[i], condition=label[i])

plt.ylabel("Success Rate", fontsize=25)
plt.xlabel("Iteration Number", fontsize=25)
plt.title("Awesome Robot Performance", fontsize=30)
plt.show()
