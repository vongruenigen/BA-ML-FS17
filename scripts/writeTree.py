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

def Tree():
    return collections.defaultdict(Tree)

def writePairs(firstValue, dict, out_f):
    for key in dict:
        if not (firstValue == "deleted" or dict[key][0] == "deleted"):
            out_f.write(firstValue + "\n")
            out_f.write(dict[key][0] + "\n")
            out_f.write(SPLIT_CONV_SYM + "\n")
        if len(dict[key][1]) > 0:
            writePairs(dict[key][0], dict[key][1], out_f)

SPLIT_CONV_SYM = '<<<<<END-CONV>>>>>'
MY_DICT_PATH = "D:/BA/raw_data/reddit_data/shelve/30wc-all-reddit2"
output_file_dir = "D:/BA/raw_data/reddit_data/reddit_30wc-all-reddit_2011_2.txt"
data_dict = shelve.open(MY_DICT_PATH, "c", writeback=True)
print('Start with generating pairs')
with open(output_file_dir, 'w+', encoding='utf8') as out_f:
    i = 0
    trees = len(data_dict)
    start_time = time.time()
    for key in data_dict:
        if len(data_dict[key][1]) > 0:
            writePairs(data_dict[key][0], data_dict[key][1], out_f)
        i = i + 1
        if (i-1) % 10**3 == 0:
            print('Processed %i tress from total  %i... (took: %.2fs)' % (
                i+1, trees, (time.time() - start_time)
            ))
            start_time = time.time()
            data_dict.sync()
print('successfully completed')
data_dict.close()