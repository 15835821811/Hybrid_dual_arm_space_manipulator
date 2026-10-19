"""Read-only delivery status, manifest verification and Chinese report."""
import argparse
import json
from pathlib import Path
from ff_prior.storage import read_json, file_digest
from .build import OUT, ROOT


def check():
    packet = read_json(OUT / 'packet.json')
    bindings = packet['inputs'] + packet['outputs']
    completion = ROOT / 'ff_prior_delivery_release/evidence/completion.json'
    if completion.exists():
        bindings += read_json(completion)['release_products']
    mismatches = [b['path'] for b in bindings
                  if not (ROOT / b['path']).is_file() or file_digest(ROOT / b['path']) != b['sha256']]
    return {'status': 'PASS' if not mismatches else 'FAIL', 'mismatches': mismatches,
            'checks': len(bindings),
            'scope': 'file integrity only; no training/physics/replay rerun'}


def status():
    completion = ROOT / 'ff_prior_delivery_release/evidence/completion.json'
    return {'engineering_delivery': read_json(completion) if completion.exists() else 'INDEPENDENT_REVIEW_PENDING',
            'packet': str(OUT / 'packet.json'),
            'TEST': read_json(ROOT / 'ff_prior_delivery_release/evidence/TEST_readiness.json'),
            'next_256': 'PROTOCOL_ONLY_NOT_STARTED', 'query_does_not_launch_work': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('status', 'report', 'check'))
    a = p.parse_args().action
    if a == 'report':
        print((ROOT / 'ff_prior_delivery_release/docs/ANALYSIS.md').read_text(encoding='utf-8'))
    else:
        result = check() if a == 'check' else status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get('status') == 'FAIL':
            raise SystemExit(1)
