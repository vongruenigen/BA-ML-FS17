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

if len(argv) < 2:
    print('ERROR: Required embeddings argument is missing')
    print('       (e.g. python scripts/extract_w2v_vocabulary.py in/w2v_embeddings out/vocab.pickle')
    sys.exit(2)

emb_path = argv[0]
voc_path = argv[1]
w2v_embs = Word2Vec.load(emb_path)
voc_out = {}

for word, entry in w2v_embs.vocab.items():
    voc_out[word] = entry.index

with open(voc_path, 'wb') as f:
    pickle.dump(voc_out, f)

print('Successfully extract the vocabulary of the embeddings %s' % emb_path)
print('Stored the generated pickle file at %s' % voc_path)
