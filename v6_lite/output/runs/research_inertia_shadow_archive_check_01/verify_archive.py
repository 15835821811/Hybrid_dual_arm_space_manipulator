"""Seal finite shadow evidence, including failed and interrupted attempts, in Git."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'v6_lite/output/runs'
SOURCE_HEAD = '19ad85c7db81a2e254fd176077b6aae39333a439'
DIRECTORIES = (
    'research_inertia_shadow_01', 'research_inertia_shadow_resumed_01',
    'research_inertia_shadow_smoke_01', 'research_inertia_shadow_tests_01',
    'research_inertia_shadow_tool_draft_01', 'research_inertia_shadow_resume_tool_01',
    'research_inertia_shadow_analysis_01', 'research_inertia_shadow_independent_audit_01',
    'research_inertia_shadow_independent_audit_02', 'research_inertia_shadow_independent_audit_03',
    'research_inertia_shadow_evidence_audit_01', 'research_inertia_shadow_evidence_audit_02',
    'research_supplement_completion_audit_01',
)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert not (OUT / 'verification.json').exists(), 'exclusive archive check already completed'
    checks, artifacts = [], []
    cached = {}

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})

    def file_hash(p):
        if p not in cached:
            cached[p] = sha(p)
        return cached[p]

    def verify_map(base, filename, exact=False):
        manifest = read(base / filename)
        for name, record in manifest.items():
            expected = record if isinstance(record, str) else record['sha256']
            p = base / name
            check(f'{base.name}/{filename}:{name}:manifest_sha', p.is_file() and file_hash(p) == expected)
            if isinstance(record, dict) and 'bytes' in record:
                check(f'{base.name}/{filename}:{name}:manifest_bytes', p.is_file() and p.stat().st_size == record['bytes'])
        if exact:
            actual = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
            check(f'{base.name}/{filename}:complete_file_set', actual == set(manifest) | {filename})

    verify_map(RUNS / 'research_inertia_shadow_01', 'replay_manifest.json')
    verify_map(RUNS / 'research_inertia_shadow_smoke_01', 'replay_manifest.json', True)
    resumed = RUNS / 'research_inertia_shadow_resumed_01'
    verify_map(resumed, 'manifest.json', True)
    check('consolidated_manifest_external_pin', file_hash(resumed / 'manifest.json') ==
          'ee98249e6ba936278e005943dc942724e57fb5ef2f59d2fe226b0e101d89158e')
    original = RUNS / 'research_inertia_shadow_01'
    inv = read(resumed / 'interrupted_source_inventory.json')
    check('interrupted_original_exact_139_file_set', len(inv) == 139 and
          {p.relative_to(original).as_posix() for p in original.rglob('*') if p.is_file()} == set(inv))
    for name, record in inv.items():
        p = original / name
        check('interrupted_original:' + name, file_hash(p) == record['sha256'] and p.stat().st_size == record['bytes'])
    check('interrupted_original_not_relabelled_complete', not (original / 'report.json').exists()
          and not (original / 'manifest.json').exists() and read(original / 'goal_pause_receipt.json')['observer_terminal_exit_code'] == 1)
    # Auditor source manifests bind repository/asset input files, not artifact maps.
    for directory in DIRECTORIES:
        base = RUNS / directory
        assert base.is_dir(), 'required artifact directory missing: ' + directory
        if directory not in ('research_inertia_shadow_01', 'research_inertia_shadow_resumed_01', 'research_inertia_shadow_smoke_01'):
            for m in sorted(base.rglob('*manifest.json')):
                if m.name not in ('source_manifest.json', 'frozen_source_manifest.json',
                                  'revision_01_source_manifest.json'):
                    verify_map(m.parent, m.name)
        for p in sorted(base.rglob('*')):
            if not p.is_file():
                continue
            name = p.relative_to(ROOT).as_posix()
            raw = p.read_bytes()
            staged = subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
            disk_sha = file_hash(p)
            check(name + ':staged_git_raw_bytes', sha_bytes(staged) == disk_sha and len(staged) == len(raw))
            artifacts.append({'path': name, 'sha256': disk_sha, 'bytes': len(raw), 'staged_git_sha256': sha_bytes(staged)})
    baseline = json.loads(subprocess.check_output(['git', 'show', SOURCE_HEAD + ':v6_lite/controller_status.json'], cwd=ROOT).decode('utf-8'))
    current = read(ROOT / 'v6_lite/controller_status.json')
    check('all_57_historical_status_values_preserved', len(baseline) == 57 and all(current.get(k) == v for k, v in baseline.items()))
    report = {
        'schema': 'finite_inertia_shadow_disk_and_staged_git_verification_v1',
        'evidence_valid': all(x['passed'] for x in checks),
        'frozen_producer_commit': SOURCE_HEAD,
        'archive_parent_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'raw_artifact_count': len(artifacts), 'raw_artifact_bytes': sum(x['bytes'] for x in artifacts),
        'check_count': len(checks), 'failed_checks': [x for x in checks if not x['passed']],
        'checks': checks, 'raw_artifacts': artifacts,
        'scope': 'Disk and index identity of all listed finite-study files including original partial observation, goal pause, failed auditor and migration. No physics or geometry rerun. Excludes this verifier and its own report; their bytes are separately checked after commit.',
        'shadow_runtime_certified': False, 'old_wall_deployment_completed': False,
    }
    (OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    own = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    (OUT / 'manifest.json').write_text(json.dumps(own, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: report[k] for k in ('evidence_valid', 'raw_artifact_count', 'raw_artifact_bytes', 'check_count', 'failed_checks')}))
    return 0 if report['evidence_valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
