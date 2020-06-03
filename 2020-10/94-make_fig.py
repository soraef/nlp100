import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = list(range(1, 102, 20))
y = [0, 0.13, 0.18, 0.20, 0.23, 0.25]

plt.plot(x, y)
plt.xlabel("beam_len")
plt.ylabel("bleu")
plt.savefig("94_result")