#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This script is responsible for extracting
#              the vocabulary from word2vec embeddings.
#

import sys
import helpers
import pickle

from gensim.models import Word2Vec

helpers.expand_import_path_to_source()

argv = sys.argv[1:]
top_nth = 0

if len(argv) < 2:
    print('ERROR: Required embeddings argument is missing')
    print('       (e.g. python scripts/extract_w2v_vocabulary.py <in-w2v-emb> <out-pickle> [<top-nth-most-occuring=max>]')
    sys.exit(2)

if len(argv) == 3:
    top_nth = int(argv[2])

emb_path = argv[0]
voc_path = argv[1]

print('Loading embeddings...')

w2v_embs = Word2Vec.load(emb_path)
voc_out = {}
voc_items = list(w2v_embs.vocab.items())

print('Loaded embeddings!')
print('Extracting vocabulary... (top %d words only)' % top_nth)

if top_nth > len(voc_items) or top_nth == 0:
    top_nth = len(voc_items)
elif top_nth < len(voc_items):
    voc_items.sort(key=lambda x: x[1].count, reverse=True)

for word, entry in voc_items[0:top_nth]:
    voc_out[word] = entry.index

with open(voc_path, 'wb') as f:
    pickle.dump(voc_out, f)

print('Successfully extracted the vocabulary of the embeddings %s' % emb_path)
print('Stored the generated pickle file at %s' % voc_path)
