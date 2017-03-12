#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This script is responsible for bringing
#              the unprocessed data of the cornell movie
#              dialogue dataset into a format which can
#              be used for training.
#
# How to use:  1. Download raw data from https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
#                 Have to check the raw data format from the other OpenSubtitles versions later.
#              2. Run this file with the top lvl path of your raw data as parameter.
#              3. You will find the output file in your given path.
#
#
#

import sys
import os
import re
import helpers

helpers.expand_import_path_to_source()

from os import path
from data_loader import DataLoader

SPLIT_SYM = '+++$+++'

LINES_FILE_NAME = 'movie_lines.txt'
CONVS_FILE_NAME = 'movie_conversations.txt'
OUTPUT_FILE_NAME = 'movie_conversations_full.txt'

argv = sys.argv[1:]

if len(argv) == 0 or not path.isdir(argv[0]):
    print('ERROR: Expected the path to the directory where the movie_lines.txt ' +
          ' and the movie_conversations.txt file remain!')
    print('       (e.g. python scripts/preprocess_cornell_movie_dialogues_data.py cornell-dataset-dir/')
    sys.exit(2)

data_dir = argv[0]

all_lines = {}

file_lines = path.join(data_dir, LINES_FILE_NAME)
file_convs = path.join(data_dir, CONVS_FILE_NAME)

output_file = path.join(data_dir, OUTPUT_FILE_NAME)

for line in open(file_lines, 'r'):
    line_parts = line.split(SPLIT_SYM)
    line_parts = list(map(lambda x: x.strip(), line_parts))
    
    line_no = line_parts[0]
    line_txt = line_parts[-1]

    all_lines[line_no] = line_txt

with open(output_file, 'w+') as f:
    all_convs = list(open(file_convs, 'r').read().split('\n'))

    for i, line in enumerate(all_convs):
        # Skip empty lines
        if not line: continue

        line_parts = line.split(SPLIT_SYM)
        conv_parts = line_parts[-1]
        conv_parts = re.sub(r'[\[\]\'\s]', '', conv_parts).split(',')
        conv_parts = list(map(lambda x: all_lines[x], conv_parts))
        conv_parts = conv_parts + [DataLoader.SPLIT_CONV_SYM]

        f.write('\n'.join(conv_parts) + '\n')

        if (i+1) % 10000 == 0:
            print('(Processed %i of %i conversations)' % (i+1, len(all_convs)))

print('Converted all conversations and stored them in %s' % output_file) 
