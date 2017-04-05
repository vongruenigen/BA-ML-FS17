#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: Generates a list of tokens from the given
#              corpus which can be used as the vocabulary.
#

import sys
import os
import pickle

from nltk import word_tokenize

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: Mandatory arguments missing')
    print('       (e.g. python scripts/build_vocabulary_from_corpus.py <corpus-in> <voc-out>')
    sys.exit(2)

corpus_in = argv[0]
voc_out = argv[1]

vocab_dict = {}

for i, line in enumerate(open(corpus_in, 'r')):
    # Skip type declarations and end-conv symbols
    if (i == 0 and line.startswith('#')) or line.startswith('<<<'):
        continue

    words = line.lower().strip('\n').split(' ')

    for w in words:
        if w in vocab_dict:
            vocab_dict[w] = vocab_dict[w] + 1
        else:
            vocab_dict[w] = 1

    if (i+1) % 10000 == 0:
        print('Processed %i lines...' % (i+1))

words_out = []
word_freq_tuples = list(vocab_dict.items())
word_freq_tuples.sort(reverse=True, key=lambda x: x[1])
words_out = list(map(lambda x: x[0], word_freq_tuples))

with open(voc_out, 'w+') as f:
    for w in words_out:
        f.write('%s\n' % w)
