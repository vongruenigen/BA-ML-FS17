#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: Analyses how much words in the supplie corpus are covered by the
#              vocabularies supplied as parameters.
#
#

import sys
import json
import pickle

files = []
pattern = "*.gz"

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Expected the path to the corpus file and at least on vocabulary')
    print('       (python ./analyze_word_coverage.py <corpus> <json-out> <voc1,voc2,voc3,...>)')
    sys.exit(2)

corpus_path = argv[0]
stats_path = argv[1]
vocabularies = argv[2].split(',')

stats = {}

for voc_path in vocabularies:
    with open(voc_path, 'rb') as voc_f:
        vocabulary = pickle.load(voc_f)
        vocabulary_filename = voc_path.split('/')[-1]

        if vocabulary_filename in stats:
            print('Skipping vocabulary %s as it was already analyzed before!' % vocabulary_filename)
            continue

        total_unknown_words = 0
        total_known_words = 0

        with open(corpus_path, 'r') as corpus_f:
            for i, line in enumerate(corpus_f):
                words = line.split(' ') # expect the corpus to be preprocessed already

                for w in words:
                    if w in vocabulary:
                        total_known_words += 1
                    else:
                        total_unknown_words += 1

                if (i+1) % 100000 == 0:
                    print('(Analyzed %i lines...)' % (i+1))

        total_word_count = total_known_words + total_unknown_words

        stats[vocabulary_filename] = {
            'total_unknown_words': total_unknown_words,
            'total_known_words': total_known_words,
            'total_unknown_words_perc': total_unknown_words / total_word_count,
            'total_known_words_perc': total_known_words / total_word_count
        }

    print('Finished processing the vocabulary %s' % vocabulary_filename)

with open(stats_path, 'w+') as f:
    json.dump(stats, f, indent=4, sort_keys=True)

print('Analyzed the corpus and vocbularies and stored the results in %s' % stats_path)
