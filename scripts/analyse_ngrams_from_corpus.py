import os
import sys
import helpers
import operator

helpers.expand_import_path_to_source()

from tqdm import tqdm
from data_loader import DataLoader
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

argv = sys.argv[1:]

if len(argv) < 4:
    print('ERROR: Missing mandatory argument(s)')
    print('       (analyze_ngrams.py <txt-corpus> <ngram-size> <out-ngram-csv> <out-word-csv> '
          '[<mode=collocation> <metric=freq|likelihood|pmi|chi_sq> <top-n=1000>])')
    sys.exit(2)

mode = 'collocations'
top_n = 1000
metric = 'freq'

corpus_path = argv[0]
ngram_size = int(argv[1])
out_ngram_path = argv[2]
out_words_path = argv[3]
metric_fn = None

if len(argv) > 4:
    mode = argv[4]

if len(argv) > 5:
    metric = argv[5]

if len(argv) > 6:
    top_n = int(argv[6])

def generate_words(data_path):
    with open(data_path, 'r') as data_f:
        num_lines = sum(1 for _ in data_f)
        data_f.seek(0)

        for line in tqdm(data_f, total=num_lines):
            if line.strip('\n') == DataLoader.SPLIT_CONV_SYM:
                continue
            else:
                for w in line.split(): yield w

if metric == 'likelihood':
    metric_fn = BigramAssocMeasures.likelihood_ratio
elif metric == 'pmi':
    metric_fn = BigramAssocMeasures.pmi
elif metric == 'chi_sq':
    metric_fn = BigramAssocMeasures.chi_sq

if metric_fn is None and metric != 'freq':
    print('ERROR: Invalid metric "%s"' % metric)
    sys.exit(2)

bcf = BigramCollocationFinder.from_words(generate_words(corpus_path))
bcf.apply_freq_filter(3)

with open(out_words_path, 'w+') as f:
    f.write('word;frequency\n')

    word_items = bcf.word_fd.items()
    word_items = list(sorted(word_items, reverse=True, key=operator.itemgetter(1)))
    
    for word, freq in word_items:
        f.write('%s;%d\n' % (word, freq))

if metric == 'freq':
    ngram_items = bcf.ngram_fd.items()
    ngram_items = list(sorted(ngram_items, reverse=True, key=operator.itemgetter(1)))
    results = list(map(operator.itemgetter(0), ngram_items))
else:
    results = bcf.nbest(metric_fn, top_n)

with open(out_ngram_path, 'w+') as f:
    f.write('ngram;frequency\n')

    for ngram in results:
        freq = bcf.ngram_fd[ngram]
        f.write('%s;%d\n' % (' '.join(ngram), freq))

print('Stored results of the analysis in %s and %s' % (out_ngram_path, out_words_path))
