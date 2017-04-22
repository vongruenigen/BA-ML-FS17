#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: This script extracts n-grams (specific n can be supplied as arguments)
#              from the given corpus and stores it in a shelve dictionary.
#

import sys
import os
import shelve
import time

from os import path

argv = sys.argv[1:]

if len(argv) < 2 or not path.isfile(argv[0]):
    print('ERROR: Expected the path to the file to be analyzed and the n-gram dimension')
    print('       (python scripts/generate_ngrams.py <corpus-in> <shelve-dir> <ngram-dim1,ngram-dim2,...> [<continue-line-nr>])')
    sys.exit(2)

corpus_in = argv[0]
shelve_dir = argv[1]
ngram_dims = argv[2]
ngrams_dims_str = []
shelve_open_flag = 'n'
corpus_name = corpus_in.split('/')[-1].split('.')[0]

try:
    ngram_dims = list(map(int, ngram_dims.split(',')))
    ngram_dims_str = list(map(str, ngram_dims))
except ValueError:
    print('ERROR: The n-gram dimensions must be positive integers, optionally separated by a comma. (e.g. "2,3,4")')
    sys.exit(2)

if len(argv) == 4:
    try:
        continue_line = int(argv[3])
        shelve_open_flag = 'c'
    except ValueError:
        print('ERROR: Expected a line number as an optional third argument.')
        sys.exit(2)
else:
    continue_line = 0

def find_ngrams(input_list, n):
    '''Generates n-grams with the definde size n from the given input list.'''
    return zip(*[input_list[i:] for i in range(n)])

def log(msg, out_file=sys.stdout):
    '''Logs a message to the specified file (defaults to stdout).'''
    out_file.write('%s\n' % msg)
    out_file.flush()

shelve_dict_path = os.path.join(shelve_dir, corpus_name)
ngram_count = 0

with open(corpus_in, 'r') as in_f:
    with shelve.open(shelve_dict_path, shelve_open_flag) as ngram_dict:
        try:
            if continue_line > 0:
                log('Forwarding to line %i!' % continue_line)

            start_time = time.time()

            for i, line in enumerate(in_f):
                if i < continue_line:
                    continue

                words = line.split()

                for ngram_dim in ngram_dims:
                    ngram_list = find_ngrams(words, ngram_dim)

                    for ngram in ngram_list:
                        curr_item = ':'.join(ngram)

                        if curr_item in ngram_dict:
                            ngram_dict[curr_item] += 1
                        else:
                            ngram_dict[curr_item] = 1
                            ngram_count += 1

                if (i+1) % 10**6 == 0:
                    log('Processed %i lines and extracted %i n-grams... (lengths: [%s], took: %.2fs)' % (
                        i+1, ngram_count, ', '.join(ngram_dims_str), (time.time() - start_time)
                    ))
                    start_time = time.time()
                    ngram_dict.sync()

        except KeyboardInterrupt:
            print('WARNING: interrupt received, stopping...')
        finally:
            ngram_dict.sync()
            log('WARNING: Stopped on line number %d' % i)
