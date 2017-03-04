from data_loader import DataLoader
import sys
import config

argv = sys.argv[1:]
c = config.Config()
d = DataLoader(c)

max_len_conv = 0
max_len = 0

for conv in d.load_conversations(argv[0], {}):
    import pdb
    pdb.set_trace()
