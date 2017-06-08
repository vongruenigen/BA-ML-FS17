import os
import sys

from operator import itemgetter

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: generate_sentence_distribution.py <corpus-txt> <out-csv>')
    sys.exit(2)

corpus_path, out_path = argv[:2]
sentences_count = {}

with open(corpus_path, 'r') as in_f:
    for line in in_f:
        sentence = line.strip('\n').strip().lower()

        if sentence in sentences_count:
            sentences_count[sentence] += 1
        else:
            sentences_count[sentence] = 1

with open(out_path, 'w+') as out_f:
    out_f.write('Sentence;Count\n')

    sent_items = sentences_count.items()
    sent_items = list(sorted(sent_items, key=itemgetter(1), reverse=True))

    for sent, count in sent_items:
        out_f.write('%s;%d\n' % (sent, count))

print('Generated sentence statistics for %s' % corpus_path)
