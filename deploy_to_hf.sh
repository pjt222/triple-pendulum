#!/bin/bash
# Upload to HF: viewer files -> Space, simulation data -> Dataset repo.
# Reads HF_TOKEN from .env file or environment.
#
# Usage:
#   ./deploy_to_hf.sh                    # Upload viewer + all data
#   ./deploy_to_hf.sh data/sim_700.json  # Upload a single data file

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

python3 -c "
import os, sys, glob

# Load token from .env or environment
token = None
if os.path.exists('.env'):
    for line in open('.env'):
        if line.startswith('HF_TOKEN='):
            token = line.split('=', 1)[1].strip()
if not token:
    token = os.environ.get('HF_TOKEN')
if not token:
    print('Error: No HF_TOKEN found in .env or environment', file=sys.stderr)
    sys.exit(1)

from huggingface_hub import HfApi
api = HfApi(token=token)

SPACE_ID = 'pjt222/triple-pendulum'
DATASET_ID = 'pjt222/triple-pendulum-data'

args = sys.argv[1:]

if args:
    # Upload specific file(s) to dataset repo
    for filepath in args:
        filename = os.path.basename(filepath)
        print(f'Uploading {filepath} -> {DATASET_ID}/{filename}')
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=DATASET_ID,
            repo_type='dataset',
        )
        print(f'  Done.')
else:
    # Full deployment: viewer -> Space, data -> Dataset
    print('=== Uploading viewer files to Space ===')
    for local, remote in [
        ('hf-spaces/README.md', 'README.md'),
        ('docs/index.html', 'index.html'),
        ('docs/viewer-config.json', 'viewer-config.json'),
    ]:
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=SPACE_ID,
            repo_type='space',
        )
        print(f'  {local} -> {remote}')

    print('=== Uploading simulation data to Dataset ===')
    data_files = sorted(glob.glob('data/simulation_*.json'))
    if not data_files:
        print('  No simulation data found in data/')
    else:
        api.upload_folder(
            folder_path='data/',
            path_in_repo='.',
            repo_id=DATASET_ID,
            repo_type='dataset',
            allow_patterns=['simulation_*.json'],
        )
        print(f'  Uploaded {len(data_files)} files')

print('Deploy complete!')
" "$@"
