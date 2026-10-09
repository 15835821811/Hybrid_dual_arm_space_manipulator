"""Read GitHub branch/tree and selected bytes; never push, train or execute physics."""
import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[2]
REPO='15835821811/Hybrid_dual_arm_space_manipulator'
BRANCH='v6.4-c3-search-aware-closed-loop-val'
REL='v6_4/releases/search_aware_warmstart_20261008_01/'
MEDIA='v6_4/visualization/search_aware_warmstart_20261008_01/'


def cmd(argv):
    return subprocess.check_output(argv,cwd=ROOT)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    record=dict(schema='c3_github_publication_verification_v1',
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='RUNNING',
        repository=REPO,branch=BRANCH,artifact_commit=a.commit,
        algorithm_producer='9f39b42775283432eb933f63a9047c488ba22070',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remote_tree_verified=False,raw_files=[])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        remote=cmd(['git','ls-remote','origin','refs/heads/'+BRANCH]).decode().split()[0]
        assert remote==a.commit,(remote,a.commit)
        record['remote_head']=remote
        api_head=json.loads(cmd(['gh','api',f'repos/{REPO}/git/ref/heads/{BRANCH}']))
        assert api_head['object']['sha']==a.commit
        record['github_ref_head']=api_head['object']['sha']
        tree=json.loads(cmd(['gh','api',f'repos/{REPO}/git/trees/{a.commit}?recursive=1']))
        assert not tree['truncated']
        remote_blobs={row['path']:row for row in tree['tree'] if row['type']=='blob'}
        local={}
        for line in cmd(['git','ls-tree','-r','-z',a.commit]).split(b'\0'):
            if not line:continue
            meta,path=line.split(b'\t'); mode,kind,oid=meta.decode().split()
            if kind=='blob':local[path.decode()]=dict(mode=mode,sha=oid)
        assert set(local)==set(remote_blobs)
        for name,row in local.items():
            assert remote_blobs[name]['sha']==row['sha'],name
            assert remote_blobs[name]['mode']==row['mode'],name
        record.update(remote_tree_verified=True,entire_repository_blob_count=len(local),
            release_blob_count=sum(n.startswith(REL) for n in local),
            media_blob_count=sum(n.startswith(MEDIA) for n in local),
            remote_tree_sha=tree['sha'],recursive_response_truncated=False)
        assert record['release_blob_count']==4717 and record['media_blob_count']==399
        samples=['README.md',REL+'release_manifest.json',REL+'snapshot/REPORT.md',
            REL+'snapshot/models/D/checkpoint_4000.pt',REL+'snapshot/models/S/checkpoint_4000.pt',
            REL+'snapshot/closed_loop_val/actual/c3_val_minus/D4000_A/attempt/actual/traces/c3_val_minus.npz',
            MEDIA+'visualization_manifest.json',MEDIA+'index.html',
            MEDIA+'replays/c3_test0_plus_R8_A/c3_test0_plus_R8_A_continuum_focus.mp4',
            MEDIA+'figures/c3_test0_plus_R8_A_tracking_native.csv']
        for name in samples:
            # Keep gh's output textual: the Windows gh wrapper transforms raw binary
            # output and rejects some checkpoint bytes. JSON/base64 or the exact
            # API-returned download URL preserves the file's binary representation.
            metadata=json.loads(cmd(['gh','api',f'repos/{REPO}/contents/{quote(name,safe="/")}?ref={a.commit}']))
            assert metadata['sha']==local[name]['sha']
            if metadata.get('encoding')=='base64' and metadata.get('content'):
                data=base64.b64decode(metadata['content'],validate=False)
                transport='GitHub contents JSON/base64'
            else:
                url=metadata['download_url']
                assert url.startswith(f'https://raw.githubusercontent.com/{REPO}/{a.commit}/')
                with urlopen(url,timeout=120) as response:data=response.read()
                transport='Exact GitHub contents download_url, binary HTTPS'
            original=cmd(['git','show',a.commit+':'+name])
            assert data==original,name
            record['raw_files'].append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                github_url=f'https://github.com/{REPO}/blob/{a.commit}/{name}',transport=transport,status='PASS'))
            print('Verified remote bytes: '+name,flush=True)
        assert cmd(['git','ls-remote','origin','refs/heads/'+BRANCH]).decode().split()[0]==a.commit
        record.update(status='PASS',main_merge=False,force_push=False,
            publication_scope='Independent branch; complete Git blob tree equality plus ten raw byte downloads.',
            final_record_commit_note='This receipt records the artifact commit. A later receipt-only commit may add this record and the final reconciliation; final branch HEAD is checked separately after that push.')
    except Exception as exc:
        record.update(status='FAIL',error=repr(exc))
        raise
    finally:
        record['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        a.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='raw_files'},ensure_ascii=False))


if __name__=='__main__':main()
