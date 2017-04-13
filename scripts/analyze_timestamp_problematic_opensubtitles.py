#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: Analyses the timestamp problematic in the opensubtitles
#              2016 corpus. It tracks how often sudden jumps occur in
#              the timestamps between each sentences (in different categories).
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

if len(argv) < 2 or not path.isdir(argv[0]):
    print('ERROR: Expected the path to the opensubtitles directory and the analyze output file')
    print('       (python ./analyze_timestamp_problematic_opensubtitles.py <data-dir> <out-file>)')
    sys.exit(2)

data_dir = argv[0]
output_file = argv[1]

print('Starting to gather all directories and files...')
for d, _, _ in os.walk(data_dir):
    if len(files) % 1000 == 0 and len(files) != 0:
        print('(Found %i files until now)' % len(files))
    files.extend(glob.glob(os.path.join(d, pattern)))

print('Found %i files which need to analyzed!' % len(files))

sentences_count = 0
file_count = len(files)
timestamp_differences = {}

for i in range(1, 21):
    timestamp_differences[i] = 0

files = set(files)
last_end_timestamp = None

def parse_timestamp(ts):
    '''Returns the number of secons this timestamp represents.'''
    ts = ts.split(':')[0:3]

    if ':' in ts[-1]:
        ts[-1] = ts[-1].replace(':', ',')

    h, m, s = list(map(lambda x: float(x.replace(',', '.')), ts))
    return (h*3600.0)+(m*60)+s

with open(output_file, 'w+') as f:
    for i, fname in tqdm(enumerate(files), total=len(files)):
        last_end_timestamp = None # no checking accros file boundaries

        with gzip.open(fname, 'rb') as gzf:
            tree = ET.fromstring(gzf.read())

            for outter_child in tree.getchildren():
                # Skip non-sentences tags
                if outter_child.tag != 's':
                    continue

                for inner_child in  outter_child.getchildren():
                    if inner_child.tag != 'time':
                        continue

                    ts_secs = inner_child.attrib['value']
                    ts_secs = parse_timestamp(ts_secs)

                    if last_end_timestamp is not None:
                        for secs in timestamp_differences.keys():
                            if (ts_secs - last_end_timestamp) >= secs:
                                timestamp_differences[secs] += 1

                    last_end_timestamp = ts_secs
                    sentences_count += 1

                    break

results_dict = {
    'timestamp_differences': timestamp_differences,
    'sentences_count': sentences_count,
    'total_files': file_count
}

with open(output_file, 'w+') as f:
    json.dump(results_dict, f, indent=4, sort_keys=True)

print('Analyzed the whole corpus and stores the results in %s' % output_file)
