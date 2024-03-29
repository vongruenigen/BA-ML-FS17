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
from os import path
from nltk import word_tokenize

reload(sys)
sys.setdefaultencoding('utf-8')

INPUT_FILE_NAME = 'RC_2015-01'
SPLIT_CONV_SYM = '<<<<<END-CONV>>>>>'
pattern = ".bz2"
argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Expected the path to the reddit file and a subreddit tag')
    sys.exit(2)

data_dir = argv[0]

years = [int(x) for x in argv[1].split(',')]
subreddit = argv[2]

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

print('Start with filtering the reddit data using your chosen subreddit tag %s.' % subreddit)

files = []
for year in years:
    for d, _, f in os.walk(os.path.join(data_dir + str(year))):
        for curr_file in f:
            if not curr_file.endswith(pattern) and not os.path.isdir(curr_file):
                files.extend(glob.glob(os.path.join(d, curr_file)))

output_file_name = 'reddit_corpus_' + subreddit + '.txt'
temp_file_name = 'temp_reddit_' + subreddit + '.txt'

temp_dir = path.join(data_dir, temp_file_name)
output_file_dir = path.join(data_dir, output_file_name)

my_dict = {}
with open(temp_dir, 'w+') as out_f:
    prev_content = ''
    for actual_path_dir in files:
        with open(actual_path_dir, 'r') as in_f:
            for i, line in enumerate(in_f):
                try:
                    json_obj = json.loads(line)
                except ValueError:
                    continue
                tag = json_obj['subreddit']
                id = json_obj['id']
                if tag == subreddit:
                    content = json_obj['body']
                    if content == prev_content:
                        continue
                    if not content == '' and not content == '[deleted]':
                        out_f.write(line)
                        prev_content = content


print('Start with sorting the datasets based on timestamp and the link_id tag.')
with open(temp_dir, 'r') as in_f:
    line_offset = []
    offset = 0
    for i, line in enumerate(in_f):
        line_offset.append(offset)
        offset += len(line) + 1
        try:
            json_obj = json.loads(line)
        except ValueError:
            print("JSON loads error")
            continue

        link_id = json_obj['link_id']
        created_utc = json_obj['created_utc']
        my_dict.update({i: str(link_id) + str(created_utc)})

    sorted_list = sorted(my_dict.items(), key=operator.itemgetter(1))
    in_f.seek(412, 0)
    print('Start copying the sorted datasets into the output file.')

    with open(output_file_dir, 'w+') as out_f:
        prev_link_id = ''
        sample_conv = 1
        for i, keyval in enumerate(sorted_list):
            key = keyval[0]
            in_f.seek(line_offset[key], 0)
            try:
                json_obj = json.loads(in_f.readline())
            except ValueError:
                print("JSON loads error")
                continue
            sample_conv += 1

            if sample_conv > 2 and sample_conv < len(sorted_list):
                second_json_line = json_obj
                first_link_id = first_json_line['link_id']
                second_link_id = second_json_line['link_id']
                content = first_json_line['body']
                second_content = second_json_line['body']
                created_utc = first_json_line['created_utc']

                value_next_line = clean_text(second_content)
                clean_content = clean_text(content)
                if value_next_line == '' or value_next_line == ' ':
                    continue
                if not clean_content == '' and not clean_content == ' ':
                    if not first_link_id == prev_link_id and not first_link_id == second_link_id:
                        first_json_line = second_json_line
                        continue
                    elif not first_link_id == prev_link_id and not prev_link_id == '':
                        out_f.write(SPLIT_CONV_SYM + "\n")
                    
                    out_f.write(clean_content + "\n")
                    prev_link_id = first_link_id
                    first_json_line = second_json_line

                    if (sample_conv + 1) == len(sorted_list) and second_link_id == prev_link_id:
                        clean_content = clean_text(content)
                        out_f.write(clean_content + "\n")
                        out_f.write(SPLIT_CONV_SYM + "\n")
                        content = second_json_line['body']
                    elif (sample_conv + 1) == len(sorted_list):
                        out_f.write(SPLIT_CONV_SYM + "\n")
                else:
                    print("delete an empty line")
                    first_json_line = second_json_line
                    if (sample_conv + 1) == len(sorted_list):
                        out_f.write(SPLIT_CONV_SYM + "\n")
                    continue
            else:
                first_json_line = json_obj


os.remove(temp_dir)

print('Converted all conversations and stored them in %s.' % output_file_name)
