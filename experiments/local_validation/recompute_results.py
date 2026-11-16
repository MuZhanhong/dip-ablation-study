"""Extract compact experiment records safely and recompute the published analyses."""
import argparse
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
target=args.out.resolve()
if target.exists():raise SystemExit('Choose a new output directory to preserve existing records.')
with ZipFile(args.archive) as z:
    for entry in z.infolist():
        p=(target/entry.filename).resolve()
        if p!=target and target not in p.parents:raise ValueError('Unsafe archive path')
    z.extractall(target)
for script in [target/'analyze_validation.py',target/'n5-validation/analyze_n5.py']:
    subprocess.run([sys.executable,'-B',str(script)],cwd=script.parent,check=True)
print('Record audits and seed analyses completed; no training was started.')
