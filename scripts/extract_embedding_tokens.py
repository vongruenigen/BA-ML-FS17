#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: Extracts the words from word2vec embeddings
#              and writes them to textfile, one per line.
#

import sys
import os
import pickle

from gensim.models import Word2Vec

argv = sys.argv[1:]

if len(argv) != 3:
    print('ERROR: Mandatory arguments missing')
    print('       (e.g. python scripts/extract_vocabulary_tokens.py <w2v-emb> <top-nth> <vocab-out>')
    sys.exit(2)

embeddings = Word2Vec.load(argv[0])
top_nth = argv[1]
top_nth_arr = []
vocab_out = open(argv[2], 'w+')

if top_nth <= 0 or top_nth > len(embeddings.vocab):
    top_nth = len(embeddings.vocab)

if top_nth == len(embeddings.vocab):
    top_nth_arr = list(embeddings.vocab.keys())
else:
    top_nth_arr = list(embeddings.vocab.items()).sort(lambda x: x[1].count)

with open(voc_out, 'w+') as out_file:
    for tok in top_nth_arr:
        out_file.write('%s\n' % tok)
