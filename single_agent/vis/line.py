from typing import Counter
import matplotlib.pyplot as plt
import csv

y1 = []
x1 = []
with open("./result/REINFORCE.csv", "r+", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x1.append(int(row['epoch_number']))
        y1.append(float(row['average reward']))

plt.plot(
    x1,
    y1,
    label='reinforce',
    linewidth=0.5,
)

y2 = []
x2 = []
with open("./result/actor_critic.csv", "r+", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x2.append(int(row['epoch_number']))
        y2.append(float(row['average reward']))

plt.plot(
    x2,
    y2,
    label='Actor-Critic',
    linewidth=0.5,
)

y3 = []
x3 = []
with open("./result/DQN.csv", "r+", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x3.append(int(row['epoch_number']))
        y3.append(float(row['average reward']))

plt.plot(
    x3,
    y3,
    label='DQN',
    linewidth=0.5,
)

plt.ylabel('average reward')
plt.xlabel('num episodes')
# plt.yticks((0, 500, 100))
# plt.xticks((0, 2000, 20))
plt.title('CartPole-v1')
plt.legend()
plt.savefig("./vis/all.png")
plt.show()