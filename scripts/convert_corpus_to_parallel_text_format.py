#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
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

with open(out_src_path, 'w+') as src_f:
    with open(out_target_path, 'w+') as target_f:
        for i, line in enumerate(open(in_path, 'r')):
            if first_sent is None:
                first_sent = line
            elif second_sent is None:
                second_sent = line

            if (first_sent is not None and 
                second_sent is not None):
                src_f.write(first_sent)
                src_f.write(second_sent)

                target_f.write(second_sent)
                target_f.write(first_sent)

                first_sent = None
                second_sent = None

            if (i+1) % 1000 == 0:
                print('Processed %i sentences...' % (i+1))
