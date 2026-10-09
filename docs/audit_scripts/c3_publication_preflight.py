"""Read-only final byte/navigation/history checks; never executes the experiment."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = '1758e13b01735b80c5a512b81cbc1d5e47a07ec9'
REL = ROOT/'v6_4/releases/search_aware_warmstart_20261008_01'
MEDIA = ROOT/'v6_4/visualization/search_aware_warmstart_20261008_01'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8').strip()


def main():
    assert git('branch','--show-current') == 'v6.4-c3-search-aware-closed-loop-val'
    assert git('remote','get-url','origin').rstrip('/').removesuffix('.git') == 'https://github.com/15835821811/Hybrid_dual_arm_space_manipulator'
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
    release=read(REL/'release_manifest.json'); media=read(MEDIA/'visualization_manifest.json')
    assert release['stage']=='final'
    checked=[]
    def check(p,want,n):
        assert p.is_file() and p.stat().st_size==n and sha(p)==want, str(p)
        checked.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=want,bytes=n))
    for row in release['inventory']:
        if row['copied']: check(REL/'snapshot'/row['path'],row['sha256'],row['size'])
    frozen=read(REL/'frozen_source_manifest.json')['source_sha256']
    delta={}
    for name,want in frozen.items():
        p=REL/'frozen_source'/name
        check(p,want,p.stat().st_size)
        if sha(ROOT/name)!=want: delta[name]=sha(ROOT/name)
    assert delta=={'v6_4/search_effect_teacher.py':'f2fd7f699554d94fe6022d1d266fdd7fba8d6ac34a499e2ff3e06d547f5fcaab'}
    for name,row in media['files'].items(): check(MEDIA/name,row['sha256'],row['bytes'])
    for name in ['release_manifest.json','portable_paths.json','frozen_source_manifest.json','historical_evidence_inventory.json']:
        p=REL/name; check(p,sha(p),p.stat().st_size)
    p=MEDIA/'visualization_manifest.json';check(p,sha(p),p.stat().st_size)
    actual_release={p.relative_to(REL).as_posix() for p in REL.rglob('*') if p.is_file()}
    assert actual_release=={row['path'].split('search_aware_warmstart_20261008_01/',1)[1] for row in checked if row['path'].startswith('v6_4/releases/')}
    assert len(actual_release)==4717
    assert {p.relative_to(MEDIA).as_posix() for p in MEDIA.rglob('*') if p.is_file()}==set(media['files'])|{'visualization_manifest.json'}
    assert len(media['files'])==398
    largest=max(checked,key=lambda r:r['bytes'])
    assert largest['bytes']<90_000_000
    changed=git('diff','--name-only',BASE,'--').splitlines()
    historical=[p for p in changed if (p.startswith(('v6_4/releases/','v6_4/visualization/','v6_lite/','model_test/','dual_arm_space_robot_2026/'))
        and not p.startswith(('v6_4/releases/search_aware_warmstart_20261008_01/','v6_4/visualization/search_aware_warmstart_20261008_01/'))
        and p not in ['v6_4/visualization/index.html','v6_lite/visualization/index.html'])]
    assert historical==[],historical
    old=subprocess.check_output(['git','show',BASE+':README.md'],cwd=ROOT)
    current=(ROOT/'README.md').read_bytes()
    assert old.replace(b'\r\n',b'\n') in current.replace(b'\r\n',b'\n')
    links=[]
    for path in ['README.md','docs/V6_4_C3_VISUALIZATION.md','docs/V6_4_C3_MAINTENANCE.md']:
        p=ROOT/path;text=p.read_text(encoding='utf-8-sig')
        if path=='README.md':text=text.split('以下保留 C.2')[0]
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            resolved=(p.parent/target.split('#')[0]).resolve()
            # The reconciliation itself is generated immediately after this receipt.
            if resolved.name=='C3_FINAL_COMPLETION_AUDIT.md' and not resolved.exists():continue
            assert resolved.exists(),(path,target)
            links.append(dict(page=path,target=target))
    report=dict(schema='c3_publication_preflight_v1',status='PASS',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)),base=BASE,local_head_at_check=git('rev-parse','HEAD'),
        branch=git('branch','--show-current'),release_manifest_sha256=sha(REL/'release_manifest.json'),
        media_manifest_sha256=sha(MEDIA/'visualization_manifest.json'),
        exact_release_files=4717,exact_media_files=399,verified_files=len(checked),
        total_bytes=sum(x['bytes'] for x in checked),largest=largest,
        frozen_sources=len(frozen),publication_source_delta=delta,historical_files_modified=[],
        historical_README_bytes_retained=True,current_navigation_links_checked=links,
        checked=checked,new_physics_steps=0,new_model_samples=0,new_training_updates=0)
    out=ROOT/'docs/audit_receipts/c3_publication_preflight.json'
    assert not out.exists(),'Preserve previous receipt; choose a new audit if checks change'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checked','current_navigation_links_checked']},ensure_ascii=False))


if __name__=='__main__':main()
