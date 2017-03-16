#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: Extracts the words from a pickle vocabulary
#              and writes them to textfile, one per line.
#

import sys
import os
import pickle

argv = sys.argv[1:]

if len(argv) != 2:
    print('ERROR: Mandatory arguments missing')
    print('       (e.g. python scripts/extract_vocabulary_tokens.py <out-txt> <vocab>')
    sys.exit(2)

out_path = argv[0]
vocabulary = pickle.load(open(argv[1], 'rb'))

with open(out_path, 'w+') as out_file:
    for i, word in enumerate(vocabulary.keys()):
        out_file.write('%s\n' % word)
