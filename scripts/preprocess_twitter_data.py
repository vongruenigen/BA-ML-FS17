#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#

import sys
import os
import pickle
import re

from nltk import word_tokenize

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: Mandatory arguments missing')
    print('       (e.g. python scripts/preprocess_twitter_data.py <raw-corpus-in> <corpus-out>')
    sys.exit(2)

raw_corpus_in = argv[0]
corpus_out = argv[1]

http_regex = re.compile(r"http\S+")
regex = re.compile('\{.+?\}')
allowed_chars = re.compile('[^\w , . ! ?]')

def clean_text(t):
    t = t.replace(str('\r\''), '')
    t = http_regex.sub('', t)
    t = ' '.join(word_tokenize(t))
    t = allowed_chars.sub('', t)
    t = t.strip('-')
    t = t.lstrip()
    t = t.strip('[')
    t = t.strip(']')
    t = t.lower()
    t = t.strip('\"')
    t = regex.sub('', t)
    t = t.replace("~", "")
    t = t.strip(' ')
    t = t.replace('...', '')
    t = t.replace('#', '')
    t = t.replace('&gt', '')
    t = t.replace('\r', '')
    t = t.replace('\n', '')
    t = t.replace('  ', ' ')
    t = re.sub(' +',' ',t)
    return t

with open(corpus_out, 'w+') as out_f:
    lines = []

    for i, line in enumerate(open(raw_corpus_in, 'r')):
        lines.append(line)

        if len(lines) < 2: continue

        for l in lines:
            l = clean_text(l)
            out_f.write('%s\n' % l)

        out_f.write('<<<<<END-CONV>>>>>\n')
        lines = []

        if (i+1) % 10000 == 0:
            print('Processed %i lines...' % (i+1))
