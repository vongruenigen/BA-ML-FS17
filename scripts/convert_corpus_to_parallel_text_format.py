#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: Creates a source and target file from a
#              a given conversational corpus to a format
#              called parallel text format for usage with:
#              https://github.com/google/seq2seq
#

import sys
import os

argv = sys.argv[1:]

if len(argv) != 3:
    print('ERROR: Mandatory arguments missing')
    print('       (e.g. python scripts/conver_corpus_to_parallel_text_format.py <in> <out-src> <out-target>')
    sys.exit(2)

in_path, out_src_path, out_target_path = argv

use_first_sent = True
first_sent = None
second_sent = None

last_sentence = None

with open(out_src_path, 'w+') as src_f:
    with open(out_target_path, 'w+') as target_f:
        for i, curr_sentence in enumerate(open(in_path, 'r')):
            if last_sentence is not None:
                src_f.write(last_sentence)
                target_f.write(curr_sentence)

            last_sentence = curr_sentence

            if (i+1) % 1000 == 0:
                print('Processed %i sentences...' % (i+1))
