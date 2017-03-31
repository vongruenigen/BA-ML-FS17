#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: Splits a given corpus file into train, valid and
#              test sets by the given ratios. The rations should
#              be a list of comma separated integers summing up
#              to 100 where the numbers stand for the ratio to use
#              for train, valid and test respectively. If requested,
#              the data can be split into multiple parts for more
#              flexibility if the dataset is huge.
#

import sys
import tqdm
import numpy as np

argv = sys.argv[1:]

if len(argv) < 5:
    print('ERROR: Required embeddings argument is missing')
    print('       (e.g. python scripts/split_corpus.py <corpus-in> <ratios> \\'
          '                                            <train-out> <valid-out> \\'
          '                                            <test-out> [<nb-parts=1>]')
    sys.exit(2)

corpus = argv[0]
ratios = [int(x) for x in argv[1].split(',')]

train_out = argv[2]
valid_out = argv[3]
test_out = argv[4]
nb_parts = 1

if len(argv) > 5:
    nb_parts = int(argv[5])

if len(ratios) != 3 or sum(ratios) != 100:
    print('ERROR: Invalid ratios (%s)' % argv[1])
    sys.exit(2)

SPLIT_CONV_SYM = '<<<<<END-CONV>>>>>'

# Count the lines of the corpus and calculate the number of
# lines for each output file
num_lines = sum(1 for line in open(corpus, 'r'))
num_train = int(num_lines * (ratios[0] / 100.0)) + 1
num_valid = int(num_lines * (ratios[1] / 100.0))
num_test  = int(num_lines * (ratios[2] / 100.0))

print('num_train=%i, num_valid=%i, num_test=%i' % (num_train, num_valid, num_test))

free_idxs = list(range(num_lines))
corpus_f = open(corpus, 'r')

# Open all files and build
train_fs, valid_fs, test_fs = [], [], []

ds_dict = {'train_out': (train_fs, 0, num_train),
           'valid_out': (valid_fs, 0, num_valid),
           'test_out':  (test_fs, 0, num_test)}

def convert_path(p, n):
    str_n = str(n)

    if p.rfind('.'):
        return '%s.%i.%s' % (p[0:p.rindex('.')], n,
                             p[p.rindex('.')+len(str_n):])
    else:
        return '%s.%i' % (p, n)

def close_files(ds, k):
    for f in ds[k][0]:
        f.close()

for i in range(nb_parts):
    train_fs.append(open(convert_path(train_out, i), 'w+'))
    valid_fs.append(open(convert_path(valid_out, i), 'w+'))
    test_fs.append(open(convert_path(test_out, i), 'w+'))

lines = []
ds_dict_keys = list(ds_dict.keys())
iterator = enumerate(corpus_f)

for i, line in tqdm.tqdm(iterator, total=num_lines):
    lines.append(line)
    line = line.strip()
    if line != (SPLIT_CONV_SYM): continue

    #lines.append(SPLIT_CONV_SYM)
    curr_idx = np.random.randint(0, len(ds_dict_keys))
    curr_key = ds_dict_keys[curr_idx]

    curr_fs, curr_num, curr_max = ds_dict[curr_key]
    curr_f = curr_fs.pop()

    for l in lines:
        curr_f.write(l)

    curr_fs.insert(0, curr_f)
    curr_num += len(lines)
    lines = []

    if curr_num >= curr_max:
        print('Finished dataset with %i sentences, stored at: %s' % (
              curr_max, ', '.join(map(lambda x: x.name, curr_fs))))
        close_files(ds_dict, curr_key)
        ds_dict_keys.remove(curr_key)
    else:
        ds_dict[curr_key] = (curr_fs, curr_num, curr_max)

print('Split corpus into train, valid and test sets!')
