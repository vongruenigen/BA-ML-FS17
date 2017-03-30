#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: Generates a dict mapping each word in the
#              given raw vocab to it's position in the
#              raw vocabulary
#

import sys
import os
import pickle

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: Mandatory arguments missing')
    print('       (e.g. python scripts/build_pickle_vocabulary_from_raw_vocabulary.py <raw-vocab-in> <pickle-vocab-out>')
    sys.exit(2)

voc_in = argv[0]
voc_out = argv[1]
voc_dict = {}

for i, word in enumerate(open(voc_in, 'r')):
    if len(line) == 0:
        continue

    voc_dict[i] = word

with open(voc_out, 'wb') as out_f:
    pickle.dump(out_f, voc_dict)

print('Successfully stored pickle vocab at %s' % voc_out)