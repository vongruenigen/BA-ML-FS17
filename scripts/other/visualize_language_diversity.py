import os
import sys
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

legend_txts = []
x_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

unigram_values = [60.4760563688, 59.994541486, 52.2903073152, 54.7835117124, 51.5633723467, 69.3818178599]
plt.plot(x_values, unigram_values)
legend_txts.append('Uni-Grams')

bigram_values = [46.7631289003, 45.2505792602, 34.305271439, 36.472859145, 33.3524195349, 40.2600133793]
plt.plot(x_values, bigram_values)
legend_txts.append('Bi-Grams')

sentence_values = [34.5654121864, 28.4218189964, 18.6371927803, 11.8183563748, 29.3918810804, 33.1209197389]
plt.plot(x_values, sentence_values)
legend_txts.append('Sentences')

gca = plt.gca()
gca.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
plt.ylim(0, 100)

plt.legend(legend_txts)
plt.show()
