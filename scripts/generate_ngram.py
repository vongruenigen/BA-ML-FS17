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
import shelve

from os import path
reload(sys)
sys.setdefaultencoding('utf-8')

argv = sys.argv[1:]
if len(argv) < 2 or not path.isfile(argv[0]):
    print('ERROR: Expected the path to the file to be analyzed and the n-gram dimension')
    sys.exit(2)

data_dir = argv[0]

_, tail = os.path.split(data_dir)
data_name = os.path.splitext(tail)[0]

try:
    ngram_dim = int(argv[1])
except ValueError:
    print('ERROR: Expected the numeric ngram dimension argument.')
    sys.exit(2)

if len(argv) == 3:
    try:
        continue_line = int(argv[2])
        selv_flag = 'c'
    except ValueError:
        print('ERROR: Expected a line number as optional second argument.')
        sys.exit(2)
else:
    continue_line = 0
    selv_flag = 'n'

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

# See https://docs.python.org/3/library/shelve.html#shelve-example

MY_DICT_PATH = "E:/BA/analyze_results/"
FILE_NAME = data_name

my_dict = os.path.join(MY_DICT_PATH, FILE_NAME)

with open(data_dir, 'r+') as in_f:
    data_dict = shelve.open(my_dict,selv_flag)
    try:
        for i, line in enumerate(in_f):
            if i < continue_line:
                continue
            words = line.split()
            ngram_list = find_ngrams(words, ngram_dim)
            for entry in ngram_list:
                cur_item = ' : '.join(map(str, entry))
                
                if cur_item in data_dict:
                    oldValue = data_dict[cur_item]
                    data_dict[cur_item] = oldValue + 1
                else:
                    data_dict[cur_item] = 1
            if (i+1) % 10**5 == 0:
                print('Generated %s entries...' % str(i+1))
                data_dict.sync()
    except KeyboardInterrupt:
        print("W: interrupt received, stopping...")
    finally:
        data_dict.sync()
        data_dict.close()
        print("Stopped on line number %d" % i)
    data_dict.close()
