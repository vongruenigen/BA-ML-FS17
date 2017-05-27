#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: This script is responsible for bringing
#              the unprocessed data of the OpenSubtitles
#              dialogue dataset into a format which can
#              be used for training.
#
# How to use:  1. Download raw data from https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/
#                 You need a Torrent for the download.
#              2. Unzip the file RC_2015-01
#              3. Run this file with the top lvl path of your RC_2015-01 file AND as second parameter you can select which dataset year you wish. As a third parameter 
#                 you select the subreddit corpus  you like.
#                 Example: ./script/preprocess_reddit_corpus ../../reddit_corpus/ 2014,2015 movies

import sys
import os
import re
import helpers
import gzip
import glob
import json
import operator
import nltk
import shelve
import collections
import time
from os import path
from nltk import word_tokenize
from imp import reload

#reload(sys)
#sys.setdefaultencoding('utf-8')

SPLIT_CONV_SYM = '<<<<<END-CONV>>>>>'

pattern = ".bz2"
selv_flag = 'n'
argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: Expected the path to the reddit file')
    sys.exit(2)

data_dir = argv[0]

years = [int(x) for x in argv[1].split(',')]
MY_DICT_PATH = "D:/BA/raw_data/reddit_data/shelve/30wc-all-reddit"

subreddit_regex = re.compile('\/[A-Za-z]{1}\/')
http_regex = re.compile(r"http\S+")
regex = re.compile('\{.+?\}')
allowed_chars = re.compile('[^\w , . ! ?]')

def clean_text(t):

    t = subreddit_regex.sub('', t)
    t = t.replace('[spoilers]', '')
    t = t.replace(str('\r\''), '')
    t = http_regex.sub('', t)
    t = ' '.join(word_tokenize(t))
    t = allowed_chars.sub('', t)
    t = t.strip('-')
    t = t.lstrip()
    t = t.strip('[')
    t = t.strip(']')
    t = t.lower()
    t = t.strip('\"')
    t = regex.sub('', t)
    t = t.replace("~", "")
    t = t.strip(' ')
    t = t.replace('...', '')
    t = t.replace('#', '')
    t = t.replace('&gt', '')
    t = t.replace('\r', '')
    t = t.replace('\n', '')
    t = t.replace('  ', ' ')
    t = re.sub(' +',' ',t)
    return t

def Tree():
    return collections.defaultdict(Tree)

def writePairs(firstValue, dict, out_f):
    for key in dict:
        out_f.write(firstValue + "\n")
        out_f.write(dict[key][0] + "\n")
        out_f.write(SPLIT_CONV_SYM + "\n")
        if len(dict[key][1]) > 0:
            writePairs(dict[key][0], dict[key][1], out_f)

def addValue(dict, nameID, parentID, clearConten):
    for key in dict:
        if key == parentID:
            dict[key][1][nameID] = [clearConten, Tree()]
            break
        else:
            addValue(dict[key][1], nameID, parentID, clearConten)


print('Start with filtering the reddit corpus. Remove lines longer 30 words.')

files = []
for year in years:
    for d, _, f in os.walk(os.path.join(data_dir + str(year))):
        for curr_file in f:
            if not curr_file.endswith(pattern) and not os.path.isdir(curr_file):
                files.extend(glob.glob(os.path.join(d, curr_file)))

output_file_name = 'reddit_corpus_wc30.txt'
output_file_dir = path.join(data_dir, output_file_name)

print('Start with creating trees')

data_dict = shelve.open(MY_DICT_PATH, "n", writeback=True)
prev_content = ''
for actual_path_dir in files:
    print("Building Tree with the File %s" % actual_path_dir)
    with open(actual_path_dir, 'r', encoding='utf8') as in_f:
        start_time = time.time()
        for i, line in enumerate(in_f):
            try:
                json_obj = json.loads(line)
            except ValueError:
                continue
            try:
                link_id = json_obj['link_id']
                parent_id = json_obj['parent_id']
                name_id = json_obj['name']
                content = json_obj['body']
            except TypeError:
                continue
            wc = len(content.split())
            if wc <= 30:
                clear_content = clean_text(content)
                if clear_content == prev_content:
                    continue
                if not clear_content == '' and not clear_content == 'deleted':
                    if link_id not in data_dict:
                        if link_id == parent_id:
                            data_dict[link_id] = [clear_content, Tree(), name_id]
                        else:
                            continue
                    else:
                        if link_id == parent_id or parent_id == data_dict[link_id][2]:
                            data_dict[link_id][1][name_id] = [clear_content, Tree()]
                        else:
                            addValue(data_dict[link_id][1], name_id, parent_id, clear_content)
                    prev_content = clear_content
            if (i+1) % 10**6 == 0:
                print('Processed %i lines... (took: %.2fs)' % (
                    i+1, (time.time() - start_time)
                ))
                start_time = time.time()
                data_dict.sync()
print('Tree completely created. Start with Output...')

with open(output_file_dir, 'w+', encoding='utf8') as out_f:
    i = 0
    trees = len(data_dict)
    start_time = time.time()
    for key in data_dict:
        if len(data_dict[key][1]) > 0:
            writePairs(data_dict[key][0], data_dict[key][1], out_f)
        i = i + 1
    if (i) % 10**5 == 0:
        print('Processed %i tress from total  %i... (took: %.2fs)' % (
            i+1, trees, (time.time() - start_time)
        ))
        start_time = time.time()
        data_dict.sync()
print('successfully completed')
data_dict.close()
