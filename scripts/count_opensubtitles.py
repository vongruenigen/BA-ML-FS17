#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#

import sys
import os
import re
import gzip
import glob
import json
import xml.etree.ElementTree as ET

from tqdm import tqdm
from os import path

files = []
pattern = "*.gz"

argv = sys.argv[1:]

if len(argv) == 0 or not path.isdir(argv[0]):
    print('ERROR: Expected the path to the opensubtitles directory and the analyze output file')
    print('       (python ./count_opensubtitles_sentences.py <data-dir>)')
    sys.exit(2)

data_dir = argv[0]

print('Starting to gather all directories and files...')
for d, _, _ in os.walk(data_dir):
    if len(files) % 1000 == 0 and len(files) != 0:
        print('(Found %i files until now)' % len(files))
    files.extend(glob.glob(os.path.join(d, pattern)))

print('Found %i files which need to analyzed!' % len(files))

sentences_count = 0
files = set(files)

for i, fname in tqdm(enumerate(files), total=len(files)):
    with gzip.open(fname, 'rb') as gzf:
        tree = ET.fromstring(gzf.read())

        for i, outter_child in enumerate(tree.getchildren()):
            # Skip non-sentences tags
            if outter_child.tag != 's':
                continue

            sentences_count += 1

print('There are %d sentences in the raw corpus' % sentences_count)
