#!/usr/bin/env python

from gensim.models import Word2Vec

import sys
import itertools
import multiprocessing
import time

#
# Parameters
#
min_count = 3
size = 52
nb_epoch = 100
window = 4
nb_threads = multiprocessing.cpu_count()

#
# Argument handling
#
argv = sys.argv[1:]

if len(argv) == 0:
    print('ERROR: corpus file(s) required')
    print('       (e.g. ./generate_word_embeddings f1.txt f2.txt ...)')
    sys.exit(2)

#
# Custom iterator for memory friendly learning
#
class SentencesIter(object):
    def __init__(self, path):
        self.path = path
 
    def __iter__(self):
        count = 0
        print('reading sentences of file %s' % self.path)

        for line in open(self.path):
            yield line.split()
            count += 1

            if count % 100000 == 0:
                print('processed %d sentences' % count)

sentence_iters = []

for file in argv:
    print(file)
    sentence_iters.append(SentencesIter(file))

model = Word2Vec(itertools.chain(*sentence_iters), size=size,
                 window=10, min_count=min_count, workers=nb_threads,
                 iter=nb_epoch)

model.save('word2vec_embeddings_%s' % int(time.time()))
