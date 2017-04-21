#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: Analyses how much words in the supplie corpus are covered by the
#              vocabularies supplied as parameters.
#

import sys
import json
import pickle

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Expected the path to the corpus file and at least on vocabulary')
    print('       (python ./analyze_word_coverage.py <corpus> <json-out> <voc1,voc2,voc3,...>)')
    sys.exit(2)

corpus_path = argv[0]
stats_path = argv[1]
vocabulary_paths = argv[2].split(',')

vocabulary_stats = {}

vocabularies = {}

for voc_path in vocabulary_paths:
    with open(voc_path, 'rb') as voc_f:
        vocabulary_filename = voc_path.split('/')[-1]
        vocabularies[vocabulary_filename] = pickle.load(voc_f)

for voc_name in vocabularies.keys():
    vocabulary_stats[voc_name] = {
        'total_known_words': 0,
        'total_unknown_words': 0
    }

total_word_count = 0

with open(corpus_path, 'r') as corpus_f:
    print('Starting to analyze the corpus in regard to the supplied vocabularies...')

    for i, line in enumerate(corpus_f):
        words = line.split(' ') # expect the corpus to be preprocessed already

        for w in words:
            for voc_name, voc_dict in vocabularies.items():
                if w in voc_dict:
                    vocabulary_stats[voc_name]['total_known_words'] += 1
                else:
                    vocabulary_stats[voc_name]['total_unknown_words'] += 1

            total_word_count += 1

        if (i+1) % 100000 == 0:
            print('(Analyzed %i lines...)' % (i+1))

    print('Finished analyzing the corpus!')

for voc_name, stats in vocabulary_stats.items():
    total_unknown_words = stats['total_unknown_words']
    total_known_words = stats['total_known_words']

    vocabulary_stats[voc_name] = {
        'total_unknown_words': stats['total_unknown_words'],
        'total_known_words': stats['total_known_words'],
        'total_unknown_words_perc': total_unknown_words / total_word_count,
        'total_known_words_perc': total_known_words / total_word_count
    }

vocabulary_stats['corpus'] = {
    'name': corpus_path.split('/')[-1],
    'total_word_count': total_word_count
}

with open(stats_path, 'w+') as f:
    json.dump(vocabulary_stats, f, indent=4, sort_keys=True)

print('Analyzed the corpus and vocbularies and stored the results in %s' % stats_path)
