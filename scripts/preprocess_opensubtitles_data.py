#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
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
from nltk import word_tokenize

files = []
pattern = "*.gz"

argv = sys.argv[1:]

if len(argv) < 2 or not path.isdir(argv[0]):
    print('ERROR: Expected the path to the opensubtitles directory and the output file')
    print('       (python ./preprocess_opensubtitles_data.py <data-dir> <out-file>)')
    sys.exit(2)

data_dir = argv[0]
output_file = argv[1]

allowed_chars = re.compile('[^\w , . ! ?]')
multi_comma_chars = re.compile('\.+')
multi_whitespace = re.compile(' +')
dash_regex = re.compile('\s+-\s+')

def clean_text(t):
    orig_t = t
    t = t.lower()
    t = allowed_chars.sub('', t)
    t = multi_comma_chars.sub(',', t)
    t = multi_whitespace.sub(' ', t)
    t = dash_regex.sub(' ', t)
    t = t.strip()

    if t.endswith(','):
        t = '%s.' % t[:-1]

    return t

print('Starting to gather all directories and files...')
for d, _, _ in os.walk(data_dir):
    if len(files) % 10000 == 0:
        print('(Found %i files until now)' % len(files))
    files.extend(glob.glob(os.path.join(d, pattern)))
print('Found %i files which need to preprocessed!' % len(files))

files = set(files)
last_text = None
count_combined_sentences = 0

def write_to_file(file, line):
    text = clean_text(line)
    file.write('%s\n' % text)

with open(output_file, 'w+') as f:
    for i, fname in tqdm(enumerate(files), total=len(files)):
        with gzip.open(fname, 'rb') as gzf:
            content = ET.fromstring(gzf.read())

            triple_dot_pred = lambda t1, t2: t1.endswith('...') and t2.startswith('...')
            comma_pred = lambda t1, t2: t1.endswith(',') and t2[0].islower()

            # We've to write out the last sentence from the last file in case that
            # there were an even number of lines, otherwise drop it
            if last_text is not None and not last_text.startswith('...') and texts_count % 2 != 0:
                write_to_file(f, last_text)

            texts_count = 0
            last_text = None

            for child in content.getchildren():
                if child.tag != 's':
                    continue

                words = []

                for node in child.getchildren():
                    if node.tag == 'w':
                        words.append(node.text)

                text = ' '.join(words)

                if text == last_text or len(words) < 2:
                    continue # skip double sentences and with less than 2 words

                if last_text is not None:
                    if triple_dot_pred(last_text, text) or comma_pred(last_text, text):
                        last_text += text
                        count_combined_sentences += 1
                    else:
                        write_to_file(f, last_text)
                        texts_count += 1
                        last_text = text
                else:
                    last_text = text

print('Converted all conversational text and stored it in %s (with %i combined sentences)' % (output_file, count_combined_sentences))
