#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#

import sys
import os
import pickle
import numpy as np

argv = sys.argv[1:]

if len(argv) != 6:
    print('ERROR: Mandatory arguments missing')
    print('       (e.g. python scripts/generate_random_integer_corpus.py <num-entries> <n-min> <n-max> <n-min-length> <n-max-length> <out-txt>')
    sys.exit(2)

num_entries = int(argv[0])
n_min = int(argv[1])
n_max = int(argv[2])
n_min_length = int(argv[3])
n_max_length = int(argv[4])
out_path = argv[5]
end_conv_token = '<<<<<END-CONV>>>>>\n'

entries = []

for i in range(num_entries):
    curr_entry = []
    curr_length = np.random.randint(n_min_length, n_max_length)

    for _ in range(curr_length):
        curr_entry.append(np.random.randint(n_min, n_max))

    entries.append(curr_entry)

with open(out_path, 'w+') as f:
    for e in entries:
        text = ' '.join(map(str, e))
        f.write('%s\n' % text)
        f.write('%s\n' % text)
        f.write(end_conv_token)

print('Stored %i random entries at %s' % (len(entries), out_path))