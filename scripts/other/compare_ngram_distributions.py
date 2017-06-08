import os
import sys
import matplotlib.pyplot as plt
import numpy as np

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: compare_ngram_distributions.py <exp-ngram-csv> <out-ngram-csv> <top-n>')
    sys.exit(2)

exp_ngram_csv, out_ngram_csv, top_n = argv[:3]
top_n = int(top_n)

out_top_ngram = []
out_top_ngram_freq = []

exp_top_ngram = []
exp_top_ngram_freq = []

ngram_name = 'Bigram'

# Colllect the top-n ngrams from the out-ngram-csv
for i, line in enumerate(open(out_ngram_csv, 'r')):
    if i == 0: continue # skip headings

    ngram, freq = line.strip('\n').split(';')

    out_top_ngram.append(ngram)
    out_top_ngram_freq.append(int(freq))

    if len(out_top_ngram) == top_n:
        break

ngram_dim = len(out_top_ngram[0])

for i, line in enumerate(open(exp_ngram_csv, 'r')):
    if i == 0: continue # skip headings

    ngram, freq = line.strip('\n').split(';')

    if ngram in out_top_ngram:
        exp_top_ngram.append(ngram)
        exp_top_ngram_freq.append(int(freq))

        if len(exp_top_ngram) == len(out_top_ngram):
            break

not_existing_list = []

if len(exp_top_ngram[0].split()) == 1:
    ngram_name = 'Unigram'

# There might be the possibility that certain ngrams do not exist in the
# corpus the expected ngrams were extracted from. To fix this problem, we
# simply add the missing ngrams and set the frequency to 0. This does not
# change the distribution at all, but it allows this script to run without
# problems.
for ngram in out_top_ngram:
    if ngram not in exp_top_ngram:
        exp_top_ngram.append(ngram)
        exp_top_ngram_freq.append(0)

if len(exp_top_ngram) != len(out_top_ngram):
    print('ERROR: Not all ngrams available in both files!')
    sys.exit(2)

print('Loaded the top %d ngrams!' % top_n)

rev_freq_out = list(reversed(out_top_ngram_freq))
rev_freq_exp = list(reversed(exp_top_ngram_freq))

sum_freq_out = sum(rev_freq_out)
sum_freq_exp = sum(rev_freq_exp)

for i in range(len(rev_freq_out)):
    prev = 0.0

    if i > 0:
        prev = rev_freq_out[i-1]

    rev_freq_out[i] = (rev_freq_out[i] / sum_freq_out)

for i in range(len(rev_freq_exp)):
    prev = 0.0

    if i > 0:
        prev = rev_freq_exp[i-1]

    rev_freq_exp[i] = (rev_freq_exp[i] / sum_freq_exp)
#
# xlocs, xlabels = plt.xticks()
# xlocs = np.arange(0, 1.2, 0.2)
# plt.xticks(xlocs, xlabels)

# rev_freq_out = list(reversed(rev_freq_out))
# rev_freq_exp = list(reversed(rev_freq_exp))
x_values = list(range(1, len(rev_freq_exp)+1))
plt.ylim(0, 0.3)

plt.bar(x_values, rev_freq_out, alpha=0.5)
plt.bar(x_values, rev_freq_exp, alpha=0.5)

plt.ylabel('')
plt.legend(['Generated %s Distribution' % ngram_name,
            'Expected %s Distribution' % ngram_name], loc='best')

# plt.show()
plt.savefig('ngram_distribution_comparison.pdf', format='pdf')
