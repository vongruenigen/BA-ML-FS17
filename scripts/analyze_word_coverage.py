#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: Analyses how much words in the supplie corpus are covered by the
#              vocabularies supplied as parameters.
#

import sys
import json
import pickle
import helpers

from collections import defaultdict
from tqdm import tqdm

helpers.expand_import_path_to_source()

from data_loader import DataLoader

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Expected the path to the corpus file and at least on vocabulary')
    print('       (python ./analyze_word_coverage.py <corpus> <json-out> <voc1,voc2,voc3,...> [unique])')
    sys.exit(2)

corpus_path = argv[0]
stats_path = argv[1]
vocabulary_paths = argv[2].split(',')
end_conv_token = DataLoader.SPLIT_CONV_SYM
vocabulary_stats = {}
vocabularies = {}

unique = False

if len(argv) > 3 and argv[3]:
    unique = True
    print('Counting unique words and ignoring duplicates!')

for voc_path in vocabulary_paths:
    with open(voc_path, 'rb') as voc_f:
        vocabulary_filename = voc_path.split('/')[-1]
        vocabularies[vocabulary_filename] = pickle.load(voc_f)

total_word_count = 0
seen_words = defaultdict(set)

max_unknown_words_per_sentence = 10
unknown_words_perc_keys = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for voc_name in vocabularies.keys():
    sentences_with_n_unknown_words = {}
    sentences_with_perc_unknown_words = {x: 0 for x in unknown_words_perc_keys}

    for i in range(0, max_unknown_words_per_sentence+1):
        sentences_with_n_unknown_words[i] = 0

    vocabulary_stats[voc_name] = {
        'total_known_words': 0,
        'total_unknown_words': 0,
        'sentences_with_n_unknown_words': sentences_with_n_unknown_words,
        'sentences_with_perc_unknown_words': sentences_with_perc_unknown_words
    }

may_consider_word = lambda w, v: not unique or (unique and w not in seen_words[v])

num_lines = sum(1 for _ in open(corpus_path, 'r'))

with open(corpus_path, 'r') as corpus_f:
    print('Starting to analyze the corpus in regard to the supplied vocabularies...')

    for i, line in tqdm(enumerate(corpus_f), total=num_lines):
        line = line.strip('\n')

        if line == end_conv_token:
            continue

        words = line.split(' ') # expect the corpus to be preprocessed already

        for j, (voc_name, voc_dict) in enumerate(vocabularies.items()):
            unknown_words_count = 0

            for w in words:
                if may_consider_word(w, voc_name):
                    seen_words[voc_name].add(w)

                    if w in voc_dict:
                        vocabulary_stats[voc_name]['total_known_words'] += 1
                    else:
                        unknown_words_count += 1
                        vocabulary_stats[voc_name]['total_unknown_words'] += 1

            sentences_with_n_unknown_words = vocabulary_stats[voc_name]['sentences_with_n_unknown_words']
            sentences_with_perc_unknown_words = vocabulary_stats[voc_name]['sentences_with_perc_unknown_words']

            if unknown_words_count >= max_unknown_words_per_sentence:
                unknown_words_count = max_unknown_words_per_sentence

            unknown_words_perc = unknown_words_count / len(words)

            if unknown_words_perc == 0:
                sentences_with_perc_unknown_words[0.0] += 1
            elif unknown_words_perc == 1.0:
                sentences_with_perc_unknown_words[1.0] += 1
            else:
                for perc_key in unknown_words_perc_keys[1:-1]:
                    if unknown_words_perc <= perc_key and unknown_words_perc >= (perc_key - 0.2):
                        sentences_with_perc_unknown_words[perc_key] += 1
                        break

            sentences_with_n_unknown_words[unknown_words_count] += 1
            vocabulary_stats[voc_name]['sentences_with_n_unknown_words'] = sentences_with_n_unknown_words

        total_word_count += len(words)

    print('Finished analyzing the corpus!')

for voc_name, stats in vocabulary_stats.items():
    if unique:
        total_word_count = len(seen_words[voc_name])

    for attr_name in ('total_unknown_words', 'total_known_words'):
        value = stats[attr_name]
        perc_value = value / total_word_count
        vocabulary_stats[voc_name]['%s_perc' % attr_name] = perc_value

vocabulary_stats['unique'] = unique
vocabulary_stats['corpus'] = {
    'name': corpus_path.split('/')[-1],
    'total_word_count': total_word_count
}

with open(stats_path, 'w+') as f:
    json.dump(vocabulary_stats, f, indent=4, sort_keys=True)

print('Analyzed the corpus and vocbularies and stored the results in %s' % stats_path)
