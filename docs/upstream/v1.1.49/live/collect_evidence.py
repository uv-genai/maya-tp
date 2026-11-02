"""Copy bounded live-acceptance artifacts; exclude source archives and model/build data."""
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path('/tmp/maya-v49-live-87')
DEST = Path('/home/ugo/software/project-maya/docs/upstream/v1.1.49/live')
DIRS = ('Maya-', 'peer-candidate-', 'peer-control-', 'cli-fixed', 'cpu-fixed', 'strict-', 'prefill-')
DEST.mkdir(parents=True, exist_ok=True)
for item in sorted(BASE.iterdir()):
    if item.is_dir():
        if item.name.startswith(DIRS):
            shutil.copytree(item, DEST / item.name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    elif item.suffix in ('.py', '.txt', '.json', '.diff') and not item.name.startswith('progress-'):
        shutil.copy2(item, DEST / item.name)
files = []
for path in sorted(DEST.rglob('*')):
    if path.is_file() and path.name != 'manifest.json':
        files.append({'path': str(path.relative_to(DEST)), 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
(DEST / 'manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(len(files), 'files,', sum(p['bytes'] for p in files), 'bytes')
