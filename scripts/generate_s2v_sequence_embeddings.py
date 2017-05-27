import os
import time
import sys
import re
import h5py
import helpers
import tempfile
import numpy as np
import tqdm

from nltk.tokenize import word_tokenize
from subprocess import call
from os import path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)
helpers.expand_import_path_to_source()

from data_loader import DataLoader

argv = sys.argv[1:]

if len(argv) < 6:
    print('ERROR: generate_sentence_embeddings.py <ft-binary> <sent2vec-model-path> <model-dim> '
          '<ngram-dim=1|2> <test-output-csv-path> <generated-sequences-h5-path> <expected-sequences-h5-path>')
    sys.exit(2)

ft_binary_path = argv[0]
s2v_model_path = argv[1]
s2v_model_dim = int(argv[2])
ngram_dim = int(argv[3])
input_seq_path = argv[4]
output_emb_path = argv[5]
expected_emb_path = argv[6]

end_conv_sym = DataLoader.SPLIT_CONV_SYM
save_size_h5 = 10**5
save_size_tkn = 10**5

if path.isfile(output_emb_path):
    os.remove(output_emb_path)

input_sequences_dir = '/'.join(input_seq_path.split('/')[:-1])
generated_outputs_path = path.join(input_sequences_dir, 'test_outputs.txt')
expected_outputs_path = path.join(input_sequences_dir, 'test_expected.txt')

print('Writting expected and generated outputs to separate files...')

with open(input_seq_path, 'r') as input_f:
    with open(generated_outputs_path, 'w+') as output_f:
        with open(expected_outputs_path, 'w+') as expected_f:
            for i, line in enumerate(input_f):
                if i == 0: continue # skip headings

                line_parts = line.split(';')
                expected = line_parts[1]
                generated = line_parts[2].strip('\n')

                output_f.write('%s\n' % generated)
                expected_f.write('%s\n' % expected)

print('Finished extracting the generated and expected outputs!')

def get_embeddings(sequences, model_path, ft_path):
    with tempfile.NamedTemporaryFile('w+') as test_f:
        with tempfile.NamedTemporaryFile('w+') as embeddings_f:
            for seq in sequences:
                test_f.write('%s\n' % ' '.join(seq))

            test_f.flush()

            command = '%s print-vectors %s < %s > %s' % (
                ft_path, model_path, test_f.name, embeddings_f.name)
            call(command, shell=True)

            embeddings = []

            for line in embeddings_f:
                line = '[%s]' % line.replace(' ', ',')
                embeddings.append(eval(line))

            assert(len(sequences) == len(embeddings))
            return np.array(embeddings)

inp_out_tuples = [(generated_outputs_path, output_emb_path),
                  (expected_outputs_path, expected_emb_path)]

for input_path, output_path in inp_out_tuples:
    with open(input_path, 'r') as input_f:
        num_lines = sum([1 for x in input_f if x != end_conv_sym])
        input_f.seek(0)
        last_start_idx = 0

        if path.isfile(output_path):
            os.remove(output_path)

        with h5py.File(output_path) as output_f:
            embeddings_ds = output_f.create_dataset('embeddings', shape=(num_lines, s2v_model_dim))
            tokenized_sequences = []
            sequences = []

            for i, line in tqdm.tqdm(enumerate(input_f), total=num_lines):
                if line.strip('\n') == end_conv_sym:
                    continue

                sequences.append(line.strip('\n'))

                if (len(sequences) % save_size_tkn == 0 and len(sequences) != 0) or i+1 == num_lines:
                    sequences = [word_tokenize(x) for x in sequences]
                    print('Preprocessed %d sequences...' % (i+1))
                    tokenized_sequences += sequences
                    sequences.clear()

                if (len(tokenized_sequences) % save_size_h5 == 0 and len(tokenized_sequences) != 0) or i+1 == num_lines:
                    embeddings = get_embeddings(tokenized_sequences, s2v_model_path, ft_binary_path)
                    end_idx = last_start_idx+len(embeddings)
                    embeddings_ds[last_start_idx:end_idx] = embeddings
                    last_start_idx = end_idx
                    tokenized_sequences.clear()
