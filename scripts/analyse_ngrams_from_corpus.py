import os
import sys
import helpers

helpers.expand_import_path_to_source()

from tqdm import tqdm
from data_loader import DataLoader
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Missing mandatory argument(s)')
    print('       (analyze_ngrams.py <txt-corpus> <ngram-size> <out-file> '
          '[<mode=collocation> <metric=likelihood|pmi|chi_sq> <top-n=1000>])')
    sys.exit(2)

mode = 'collocations'
top_n = 1000
metric = 'likelihood'

corpus_path = argv[0]
ngram_size = int(argv[1])
out_path = argv[2]
metric_fn = None

if len(argv) > 3:
    mode = argv[3]

if len(argv) > 4:
    metric = argv[4]

if len(argv) > 5:
    top_n = int(argv[5])

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

if metric_fn is None:
    print('ERROR: Invalid metric "%s"' % metric)
    sys.exit(2)

bcf = BigramCollocationFinder.from_words(generate_words(corpus_path))
bcf.apply_freq_filter(3)

results = bcf.nbest(metric_fn, top_n)

with open(out_path, 'w+') as f:
    for ngram in results:
        f.write('%s\n' % ' '.join(ngram))

print('Stored results of the analysis in %s' % out_path)
