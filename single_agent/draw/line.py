import matplotlib.pyplot as plt
import csv

y = []
count = 0
with open("./result/REINFORCE.csv", "r+", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        count += 1
        y.append(row['reward'])

x = range(0, count)

plt.plot(
    x,
    y,
    label='REINFORCE',
    linewidth=0.03,
)
plt.xlabel('reward per epoch')
plt.ylabel('epoch number')
plt.yticks((0, 100, 10))
plt.title('REINFORCE ALGO')
plt.legend()
plt.savefig("./draw/REINFORCE.png")
plt.show()