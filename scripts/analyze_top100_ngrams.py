#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: This script sorts a shelve n-gram file and prints the top 100 entries.


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
