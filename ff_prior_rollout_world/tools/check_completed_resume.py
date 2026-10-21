"""Once-only terminal resume/hash check, never replays completed physics."""
import json
import os
import subprocess
import sys
from ff_prior.storage import read_json, file_digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_rollout_world.protocol import ROOT, OUT, PACKAGE
from ff_prior_rollout_world.inputs import identity

def main():
    target = PACKAGE/'evidence/completed/completed_resume.json'
    if target.exists():
        raise RuntimeError('once-only check already recorded')
    assert read_json(OUT/'status.json')['status'] in ('FINISHED','FINISHED_WITH_BLOCKED_ITEMS')
    def snapshot():
        return {p.relative_to(OUT).as_posix(): file_digest(p)
                for p in sorted(OUT.rglob('*')) if p.is_file()}
    before = snapshot()
    atomic_json(target, {'status':'INVOKING_ONCE','before':before})
    result = subprocess.run([sys.executable, '-m', 'ff_prior_rollout_world.job', 'resume'],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                            env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    after = snapshot()
    parsed = json.loads(result.stdout)
    unchanged = {p: file_digest(ROOT/p) == h for p,h in
                 read_json(PACKAGE/'evidence/preflight.json')['original_modified_files'].items()}
    proof = dict(returncode=result.returncode, response=parsed, stderr=result.stderr,
                 before=before, after=after, core_files_unchanged=before == after,
                 frozen_identity_unchanged=identity() == read_json(OUT/'run.json')['identity'],
                 user_files_unchanged=unchanged)
    atomic_json(target, proof)
    assert result.returncode == 0 and parsed['resume'] == 'NO_WORK_ALREADY_COMPLETE'
    assert before == after and proof['frozen_identity_unchanged'] and all(unchanged.values())
    print(json.dumps({'resume':parsed['resume'],'files':len(before),'unchanged':True}))

if __name__ == '__main__': main()
