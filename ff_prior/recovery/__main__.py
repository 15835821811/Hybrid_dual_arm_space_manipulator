import argparse
import json
from . import recover_dead_lock
parser=argparse.ArgumentParser()
parser.add_argument("--run-dir",required=True)
args=parser.parse_args()
print(json.dumps(recover_dead_lock(args.run_dir),indent=2))
