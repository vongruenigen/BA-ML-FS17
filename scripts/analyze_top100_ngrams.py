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
import operator

from os import path
reload(sys)
sys.setdefaultencoding('utf-8')

argv = sys.argv[1:]
if len(argv) < 1 or not path.isfile(argv[0]):
    print('ERROR: Expected the path to the file to the n-gram which you want to analyze.')
    sys.exit(2)

data_dir = argv[0]

data_dict = shelve.open(data_dir)
sorted_data_dict = sorted(data_dict.items(), key=operator.itemgetter(1),reverse=True)
top_hundred = sorted_data_dict[:100]
print(top_hundred)
