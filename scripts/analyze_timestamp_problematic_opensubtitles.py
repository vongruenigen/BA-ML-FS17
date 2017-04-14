#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
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

for i in range(1, 31):
    timestamp_differences[i] = 0

files = set(files)
unparseable_ts_count = 0
last_end_timestamp = None
last_id = None

ts_split_regex = re.compile('[\.,:]')
num_reg = re.compile('[^0-9]')

def parse_timestamp(ts):
    '''Returns the number of secons this timestamp represents.'''
    try:
        ts = ts_split_regex.split(ts)[0:3]
        h, m, s = list(map(float, ts))
        return (h*3600.0)+(m*60)+s
    except:
        print('Unparsable timestamp: %s' % ts)
        return -1

diff_keys = list(sorted(timestamp_differences.keys(), reverse=True))

with open(output_file, 'w+') as f:
    for i, fname in tqdm(enumerate(files), total=len(files)):
        last_end_timestamp = None # no checking accros file boundaries
        last_id = None

        with gzip.open(fname, 'rb') as gzf:
            tree = ET.fromstring(gzf.read())

            for i, outter_child in enumerate(tree.getchildren()):
                # Skip non-sentences tags
                if outter_child.tag != 's':
                    continue

                word_count = 0

                for inner_child in  outter_child.getchildren():
                    if inner_child.tag != 'time':
                        continue

                    ts_secs = inner_child.attrib['value']
                    ts_secs = parse_timestamp(ts_secs)

                    if ts_secs == -1:
                        unparseable_ts_count += 1
                        last_end_timestamp = None
                        last_id = None
                        break

                    if last_end_timestamp is not None:
                        for secs in diff_keys:
                            if (ts_secs - last_end_timestamp) >= secs:
                                timestamp_differences[secs] += 1
                                break

                    last_end_timestamp = ts_secs
                    sentences_count += 1

                    break

sum_timestamp_differences = sum(timestamp_differences.values())
timestamp_differences_perc = {x: y/sum_timestamp_differences for x, y in timestamp_differences.items()}

results_dict = {
    'timestamp_differences': timestamp_differences,
    'timestamp_differences_perc': timestamp_differences_perc,
    'sentences_count': sentences_count,
    'total_files': file_count,
    'unparseable_timestamp_count': unparseable_ts_count
}

with open(output_file, 'w+') as f:
    json.dump(results_dict, f, indent=4, sort_keys=True)

print('Analyzed the whole corpus and stores the results in %s' % output_file)
