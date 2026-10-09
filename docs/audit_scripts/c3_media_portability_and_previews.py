"""Copy finished media, verify all bytes, and prepare preview contact sheets.

This is delivery QA only. Contact sheets are pending visual review, not a PASS.
The original media and scientific run are never modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageOps


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--qa', type=Path, required=True)
    a = parser.parse_args()
    media, qa = a.media.resolve(), a.qa.resolve()
    assert not media.is_relative_to(qa) and not qa.is_relative_to(media)
    relocated, previews = qa/'relocated_media', qa/'previews_review_01'
    assert not relocated.exists() and not previews.exists()
    manifest = json.loads((media/'visualization_manifest.json').read_text())
    shutil.copytree(media, relocated)
    assert {p.relative_to(relocated).as_posix() for p in relocated.rglob('*') if p.is_file()} == set(manifest['files']) | {'visualization_manifest.json'}
    for relative, bound in manifest['files'].items():
        p = (relocated/relative).resolve()
        assert p.is_relative_to(relocated)
        assert sha(p) == bound['sha256'] and p.stat().st_size == bound['bytes']
    assert sha(relocated/'visualization_manifest.json') == sha(media/'visualization_manifest.json')
    relocation = dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),
        media_manifest_sha256=sha(media/'visualization_manifest.json'),files_verified=len(manifest['files'])+1,
        source=str(media),destination=str(relocated),browser_review='SEPARATE_REQUIRED',
        script_sha256=sha(Path(__file__)),physics_steps=0,model_samples=0)
    with (qa/'media_relocation_01.json').open('x',encoding='utf-8') as f:
        json.dump(relocation,f,indent=2)
    previews.mkdir()
    data=json.loads((media/'dashboard_data.json').read_text())
    slots=[s for s in data['slots'] if s['actual_steps']>0 and not s.get('alias_of_slot')]
    records=[]
    for offset in range(0,len(slots),4):
        sheet=Image.new('RGB',(1920,2080),'#eef2f8')
        draw=ImageDraw.Draw(sheet)
        rows=[]
        for row,s in enumerate(slots[offset:offset+4]):
            sid=s['task_id']+'_'+s['method']
            paths=[media/'replays'/sid/(sid+'_'+suffix+'.png') for suffix in ['five_view_preview','continuum_focus_preview','last_saved_state']]
            draw.text((8,row*520+8),sid+' | grid / continuum focus / last saved state | REVIEW PENDING',fill='black')
            for col,p in enumerate(paths):
                assert p.exists()
                with Image.open(p) as im:
                    thumb=ImageOps.contain(im.convert('RGB'),(630,480))
                    sheet.paste(thumb,(col*640+(640-thumb.width)//2,row*520+32))
            rows.append(dict(slot=sid,files=[dict(path=p.relative_to(media).as_posix(),sha256=sha(p)) for p in paths]))
        out=previews/('previews_%02d.jpg'%(offset//4+1))
        sheet.save(out,quality=95)
        records.append(dict(path=out.name,sha256=sha(out),rows=rows))
    with (previews/'inventory.json').open('x',encoding='utf-8') as f:
        json.dump(dict(status='VISUAL_REVIEW_PENDING',source_manifest_sha256=sha(media/'visualization_manifest.json'),sheets=records),f,indent=2)
    print(json.dumps(dict(relocation=relocation,preview_sheets=len(records),preview_pngs=len(slots)*3)))


if __name__=='__main__':
    main()
