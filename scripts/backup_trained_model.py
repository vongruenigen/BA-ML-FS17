#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: Backups the newest model in the given directory to the
#              specified backup location.
#

import sys
import json
import zipfile

REQUIRED_FILES_ENDINGS = ['config.json', 'metrics.json', 'checkpoint',
                          '.data-00000-of-00001', '.index', '.meta']

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: Missing mandatory parameters!')
    print('       (e.g. python scripts/backup_trained_model.py <model-path> <backup-dir>)')
    sys.exit(2)
