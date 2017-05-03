from matplotlib.mlab import normpdf

import numpy as np
import pylab as p
import matplotlib.pyplot as plt


sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
    ret = []

    for v in x:
        print(v)
        if v > 0:
            ret.append(v)
        else:
            ret.append(0)

    return ret

def binary_step(x):
    ret = []

    for v in x:
        if v < 0:
            ret.append(0)
        else:
            ret.append(1)

    return ret

a = sigmoid

x = np.arange(-6, 6, 0.01)
y = a(x)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_ylim(-2, 2)
ax.set_xlim(-6, 6)

# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(x,y)
plt.show()