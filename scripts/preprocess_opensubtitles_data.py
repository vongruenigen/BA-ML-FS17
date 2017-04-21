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

from __future__ import print_function

import sys
import os
import re
import gzip
import glob
import xml.etree.ElementTree as ET
import hashlib

from tqdm import tqdm
from os import path
from nltk import word_tokenize

files = []
pattern = '*.gz'

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

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

print('Starting to gather all directories and files...')
for d, _, _ in os.walk(data_dir):
    if len(files) % 10000 == 0:
        print('(Found %i files until now)' % len(files))

    files.extend(glob.glob(os.path.join(d, pattern)))

print('Found %i files which need to preprocessed!' % len(files))

files = set(files)
last_text = None
texts_count = 0
count_combined_sentences = 0
count_total_sentences = 0
count_skipped_sentences = 0
count_skipped_files = 0
seen_document_ids = {}

def write_to_file(f, line):
    text = clean_text(line)
    f.write('%s\n' % text)

def write_if_necessary(f, line, count):
    if line is not None:
        if not line.startswith('...') and texts_count % 2 != 0:
            write_to_file(f, line)
            return True
        else:
            return False

def md5(fname):
    hash_md5 = hashlib.md5()

    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()

triple_dot_pred = lambda t1, t2: t1.endswith('...') and t2.startswith('...')
comma_pred = lambda t1, t2: t1.endswith(',') and t2[0].islower()

with open(output_file, 'w+') as f:
    for i, fname in tqdm(enumerate(sorted(files)), total=len(files)):
        with gzip.open(fname, 'rb') as gzf:
            content = ET.fromstring(gzf.read())
            content_id = content.get('id')

            if len(content_id) > 0:
                content_id = int(content_id)
            else:
                content_id = None

            if content_id is not None:
                if content_id in seen_document_ids:
                    new_doc_md5 = md5(fname)

                    if seen_document_ids[content_id] == new_doc_md5:
                        print('Skipping document %s as it was already processed before! (md5: stored=%s, new=%s)' % (
                                fname, seen_document_ids[content_id], new_doc_md5
                            )
                        )
                        count_skipped_files += 1
                        continue
                    else:
                        print('Processing document %s even though the ids are same, the md5 hashes are not! (md5: stored=%s, new=%s)' % (
                            fname, seen_document_ids[content_id], new_doc_md5
                        ))
                else:
                    seen_document_ids[content_id] = md5(fname)

            # We've to write out the last sentence from the last file in case that
            # there were an odd number of lines, otherwise drop it
            if last_text is not None and not write_if_necessary(f, last_text, texts_count):
                count_skipped_sentences += 1

            texts_count = 0
            last_text = None

            for child in content.getchildren():
                if child.tag != 's':
                    continue

                count_total_sentences += 1
                words = []

                for node in child.getchildren():
                    if node.tag == 'w':
                        words.append(node.text)

                text = ' '.join(words)

                if text == last_text or len(words) < 2:
                    count_skipped_sentences += 1
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

    if not write_if_necessary(f, last_text, texts_count):
        count_skipped_sentences += 1

print('Converted all conversational text and stored it in %s (combined=%i, skipped=%i, skipped_files=%i, total=%i)' % (
        output_file, count_combined_sentences, count_skipped_sentences, count_skipped_files, count_total_sentences
    )
)
