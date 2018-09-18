import numpy as np
from matplotlib import pyplot as plt

'''
梯度爆炸
'''
dimensionality = 10
print(1 / (0.01 ** dimensionality))

max_dimensionality = 10
ax = plt.axes(xlim=(0, max_dimensionality), ylim=(0, 1 / (0.01 ** max_dimensionality)))
x = np.linspace(0, max_dimensionality, 1000)
y = 1 / (0.01 ** x)
plt.plot(x, y, lw=2)
plt.show()
