#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: Splits a given corpus file into train, valid and
#              test sets by the given ratios. The rations should
#              be a list of comma separated integers summing up
#              to 100 where the numbers stand for the ratio to use
#              for train, valid and test respectively.
#

import sys
import tqdm
import numpy as np

argv = sys.argv[1:]

if len(argv) < 5:
    print('ERROR: Required embeddings argument is missing')
    print('       (e.g. python scripts/split_corpus.py <corpus-in> <ratios> <train-out> <valid-out> <test-out>')
    sys.exit(2)

corpus = argv[0]
ratios = [int(x) for x in argv[1].split(',')]

train_out = argv[2]
valid_out = argv[3]
test_out = argv[4]

if len(ratios) != 3 or sum(ratios) != 100:
    print('ERROR: Invalid ratios (%s)' % argv[1])
    sys.exit(2)

# Count the lines of the corpus and calculate the number of
# lines for each output file
num_lines = sum(1 for line in open(corpus, 'r'))
num_train = int(num_lines * (ratios[0] / 100.0)) + 1
num_valid = int(num_lines * (ratios[1] / 100.0))
num_test  = int(num_lines * (ratios[2] / 100.0))

print('num_train=%i, num_valid=%i, num_test=%i' % (num_train, num_valid, num_test))

free_idxs = list(range(num_lines))

with open(train_out, 'w+') as train_f:
    with open(valid_out, 'w+') as valid_f:
        with open(test_out, 'w+') as test_f:
            ds_dict = {train_out: (train_f, 0, num_train),
                       valid_out: (valid_f, 0, num_valid),
                       test_out:  (test_f, 0, num_test)}

            ds_dict_keys = list(ds_dict.keys())

            for i, line in tqdm.tqdm(enumerate(open(corpus, 'r')), total=num_lines):
                rand_idx = np.random.randint(0, len(ds_dict_keys))
                rand_key = ds_dict_keys[rand_idx]

                rand_f, rand_curr, rand_max = ds_dict[rand_key]
                
                rand_f.write(line)
                rand_curr += 1

                if rand_curr == rand_max:
                    print('Finished corpus, stored at %s' % rand_f.name)
                    ds_dict_keys.remove(rand_key)
                else:
                    ds_dict[rand_key] = (rand_f, rand_curr, rand_max)

print('Split corpus into train, valid and test sets!')
