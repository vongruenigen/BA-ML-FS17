#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
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

emb_path = argv[0]
embeddings = Word2Vec.load(emb_path)
vocab_out = argv[2]
top_nth = int(argv[1])
top_nth_arr = []

if top_nth <= 0 or top_nth > len(embeddings.vocab):
    top_nth = len(embeddings.vocab)

top_nth_arr = list(embeddings.vocab.items())
top_nth_arr.sort(reverse=True, key=lambda x: x[1].count)
top_nth_arr = top_nth_arr[:top_nth]

# import pdb
# pdb.set_trace()

with open(vocab_out, 'w+') as out_file:
    for tok in map(lambda x: x[0], top_nth_arr):
        out_file.write('%s\n' % tok)

print('Extracted %i words from embeddings %s and stored at %s' % (len(top_nth_arr), emb_path, vocab_out))

