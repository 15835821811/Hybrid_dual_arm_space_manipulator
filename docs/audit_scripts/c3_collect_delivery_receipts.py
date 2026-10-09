"""Copy completed delivery receipts without altering their original bytes."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    selected = [p for p in a.source.iterdir() if p.is_file()
                and p.suffix in {'.json', '.jpg'} and not p.name.startswith('http')]
    directories = ['figures_review_01', 'frozen_load_command_01',
        'frozen_producer_load_01', 'media_input_01', 'media_portability_command_01',
        'media_render_01', 'numeric_qa_command_01', 'portable_load_command_01',
        'previews_review_01', 'publication_plan_01', 'release_export_01',
        'seed_binding_maintenance_01', 'video_qa_command_01', 'video_review_01']
    for name in directories:
        selected.extend(p for p in (a.source / name).iterdir()
                        if p.is_file() and p.suffix in {'.json', '.txt', '.xml', '.jpg', '.py', '.paths'})
    inventory = []
    for src in sorted(selected):
        rel = src.relative_to(a.source)
        dst = a.output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        original = hashlib.sha256(src.read_bytes()).hexdigest()
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == original
        inventory.append(dict(path=rel.as_posix(), original=str(src.resolve()),
                              bytes=dst.stat().st_size, sha256=original))
    record = dict(schema='c3_delivery_receipt_copy_v1', status='PASS',
        utc=dt.datetime.now(dt.timezone.utc).isoformat(), source=str(a.source.resolve()),
        copy_policy='Exact original receipts/logs/contact sheets. Excludes relocated release/media, frozen-source stage, extracted full frames, test tmp and HTTP logs.',
        original_pending_receipts='Machine-only and in-progress receipts retain their original statuses; later visual/browser verdicts are separate files.',
        inventory=inventory, count=len(inventory), total_bytes=sum(x['bytes'] for x in inventory),
        new_physics_steps=0, new_model_samples=0, new_training_updates=0)
    (a.output / 'copy_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k != 'inventory'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
