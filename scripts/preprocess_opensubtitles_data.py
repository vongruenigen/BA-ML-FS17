#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This script is responsible for bringing
#              the unprocessed data of the OpenSubtitles
#              dialogue dataset into a format which can
#              be used for training.
#
# How to use:  1. Download raw data via the following command:
#                 > wget http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz -O opensubtitles.tar.gz
#                 Have to check the raw data format from the other OpenSubtitles versions later.
#              2. Run this file with the top lvl path of your raw data as parameter.
#              3. You will find the output file in your given path.
#
#
#

import sys
import os
import re
import gzip
import glob
import xml.etree.ElementTree as ET

from tqdm import tqdm
from os import path

files = []
pattern = "*.gz"

argv = sys.argv[1:]

if len(argv) < 2 or not path.isdir(argv[0]):
    print('ERROR: Expected the path to the opensubtitles directory and the output file')
    print('       (python ./preprocess_opensubtitles_data.py <data-dir> <out-file>)')
    sys.exit(2)

data_dir = argv[0]
output_file = argv[1]

def clean_text(t):
    t = t.strip('-')
    t = t.lower()
    t = t.strip('\"')
    regex = re.compile('\(.+?\)')
    t = regex.sub('', t)
    t.replace('  ', ' ')
    regex = re.compile('\{.+?\}')
    t = regex.sub('', t)
    t = t.replace('  ', ' ')
    t = t.replace("~", "")
    t = t.strip(' ')
    t = t.replace('...', '')
    t = t.replace('#', '')

    return t

print('Starting to gather all directories and files...')
for d, _, _ in os.walk(data_dir):
    if len(files) % 1000 == 0:
        print('(Found %i files until now)' % len(files))
    files.extend(glob.glob(os.path.join(d, pattern)))
print('Found %i files which need to preprocessed!' % len(files))

with open(output_file, 'w+') as f:
    # Write the header to indicate the type of data
    f.write('# type=only-sentences')

    for i, fname in tqdm(enumerate(files), total=len(files)):
        with gzip.open(fname, 'rb') as gzf:
            tree = ET.fromstring(gzf.read())

            for child in tree.getchildren():
                if child.tag != 's':
                    continue

                words = []

                for node in child.getchildren():
                    if node.tag == 'w':
                        words.append(node.text.replace('-', ''))

                text = clean_text(' '.join(words))

                try:
                    if text[0] != '[' and text[-1] != ':':
                        f.write(text + "\n")
                except IndexError:
                    pass

print('Converted all conversational text and stored it in %s' % output_file)
