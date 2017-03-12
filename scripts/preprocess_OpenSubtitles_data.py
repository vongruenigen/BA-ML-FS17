#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This script is responsible for bringing
#              the unprocessed data of the OpenSubtitles
#              dialogue dataset into a format which can
#              be used for training.
#
# How to use:  1. Download raw data from http://opus.lingfil.uu.se/OpenSubtitles.php
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
import gzip
from glob import glob

helpers.expand_import_path_to_source()

from os import path
from data_loader import DataLoader

files = []
pattern = "*.gz"
OUTPUT_FILE_NAME = 'opensubtitle_conversation_full.txt'

argv = sys.argv[1:]

if len(argv) == 0 or not path.isdir(argv[0]):
    print('ERROR: Expected the path to the OpenSubtitles directory')
    sys.exit(2)

data_dir = argv[0]
output_file = path.join(data_dir, OUTPUT_FILE_NAME)

for dir,_,_ in os.walk(data_dir):
    files.extend(glob(os.path.join(dir,pattern)))

with open(output_file, 'w+') as f:
    for i, val in enumerate(files):
        for line in gzip.open(val, 'rb'):
            currentline = line.decode('utf-8').strip()
            currentline = re.sub('<[^>]*>', '', currentline)
            if currentline == '...' or currentline == '' or currentline == ' ':
                continue
            if currentline == '.' or currentline == '?' or currentline == '!':
                f.write(currentline + '\n')
            else:
                f.write(currentline + ' ')

        if (i+1) % 10000 == 0:
            print('(Processed %i of %i conversations)' % (i+1, len(files)))
        f.write('<<<<<END-CONV>>>>>' + '\n')

print('Converted all conversations and stored them in %s' % output_file) 