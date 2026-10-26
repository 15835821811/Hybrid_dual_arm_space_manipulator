"""Build only the explicitly listed static delivery; never run simulations."""
from pathlib import Path
import hashlib,json,re,shutil,sys
from urllib.parse import unquote,urlsplit
root=Path(__file__).resolve().parents[1]
dest=(root/(sys.argv[1] if len(sys.argv)>1 else "_site")).resolve()
if dest.exists():raise SystemExit("Refuse to overwrite existing site directory")
dest.mkdir(parents=True)
manifest=json.loads((root/"SNAPSHOT_PUBLICATION.json").read_text(encoding="utf-8"))
for item in manifest["site_files"]:
    src=root/item["path"]; raw=src.read_bytes()
    assert len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["sha256"],item["path"]
    target=dest/item["path"];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
shutil.copyfile(root/"SNAPSHOT_PUBLICATION.json",dest/"SNAPSHOT_PUBLICATION.json")
checks=0
for html in dest.rglob("*.html"):
    text=html.read_text(encoding="utf-8")
    for ref in re.findall(r'(?:src|href)="([^"#]+)"',text):
        url=urlsplit(ref)
        if url.scheme or url.netloc:continue
        target=(html.parent/unquote(url.path)).resolve()
        assert target.is_relative_to(dest) and target.is_file(),str(html)+": "+ref
        checks+=1
assert sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())<150*1024**2
print(json.dumps({"status":"PASS","site_files":len(manifest["site_files"])+1,"resource_checks":checks,"new_physics_steps":0}))
