#!/usr/bin/env python

import sys
import itertools
import multiprocessing
import time
import fasttext

#
# Parameters
#
params = {
    lr: 0.02,
    dim: 300,
    ws: 5,
    epoch: 1,
    min_count: 5,
    neg: 5,
    loss: 'ns',
    bucket: 2e6,
    minn: 3,
    maxn: 6,
    thread: multiprocessing.cpu_count(),
    t: 1e-4,
    lr_update_rate: 100
}

#
# Argument handling
#
argv = sys.argv[1:]

if len(argv) == 0:
    print('ERROR: corpus file(s) required')
    print('       (e.g. ./generate_word_embeddings f1.txt f2.txt ...)')
    sys.exit(2)

embeddings = fasttext.cbow()
