#!/usr/bin/env python

import sys
import time
import json
import math

from os import path

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: vocabulary or corpus files missing!')
    print('       (python scripts/generate_corpus_statistics.py <vocabulary> <stats-out> <corpora-in>)')
    sys.exit(2)

vocabulary_path = argv[0]
out_path = argv[1]
data_paths = argv[2:]
vocabulary = None

stats = {'data': {}}
stats_vocab = {}

with open(vocabulary_path, 'rb') as f:
    vocabulary = [line.decode('utf-8').strip('\n') for line in f]

for file_path in data_paths:
    print('Starting to process file %s' % file_path)

    sentence_count = 0
    sentence_word_len = 0
    sentence_char_len = 0
    sentence_max_word_len = -1
    sentence_min_word_len = math.inf

    words_missing_in_vocab = 0
    words_present_in_vocab = 0

    for sentence in open(file_path, 'r'):
        sent_parts = sentence.strip('\n').split(' ')
        
        for t in sent_parts:
            if t in vocabulary:
                words_present_in_vocab += 1
            else:
                words_missing_in_vocab += 1

        len_sentence = len(sent_parts)

        sentence_count += 1
        sentence_word_len += len_sentence
        sentence_char_len += sum([len(w) for w in sent_parts])

        if len_sentence > sentence_max_word_len:
            sentence_max_word_len = len_sentence

        if len_sentence < sentence_min_word_len:
            sentence_min_word_len = len_sentence

        if sentence_count % 1000 == 0:
            print('Processed %d sentences' % sentence_count)

        if sentence_count == 0:
            continue

        stats['data'].setdefault(file_path, {
            'total': {
                'max_word_count': int(sentence_max_word_len),
                'min_word_count': int(sentence_min_word_len),
                'word_count': int(sentence_word_len),
                'sentence_count': int(sentence_count),
                'character_count': int(sentence_char_len),
                'words_missing_in_vocab': int(words_missing_in_vocab),
                'words_present_in_vocab': int(words_present_in_vocab),
                'words_missing_in_vocab_percentage': float(words_missing_in_vocab / float(sentence_word_len)),
                'words_precent_in_vocab_percentage': float(words_present_in_vocab / float(sentence_word_len))
            },
            'avg': {
                'word_count': float(sentence_word_len / float(sentence_count)),
                'character_count': float(sentence_char_len / float(sentence_count)),
                'words_missing_in_vocab': float(words_missing_in_vocab / float(sentence_count)),
                'words_present_in_vocab': float(words_present_in_vocab / float(sentence_count))
            }
        })

    print('Finished analyzing the file %s' % file_path)

with open(out_path, 'w+') as f:
    f.write(json.dumps(stats, sort_keys=True,
                       indent=4, separators=(',', ': ')))

print('Successfully saved corpora statistics to %s' % out_path)