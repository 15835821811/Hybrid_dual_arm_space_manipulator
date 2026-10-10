"""Export immutable review evidence; inventory all retained local physical traces.

This does not run the controller, model, prediction, or Actual executor. Embedded
producer paths stay unchanged so the snapshot remains byte-for-byte evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--integrity", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run, out = a.run.resolve(), a.output.resolve()
    if not (run / "execution_complete.json").is_file():
        raise RuntimeError("Do not export an incomplete physical run as final")
    summary = read(a.results / "summary.json")
    integrity = read(a.integrity)
    if summary["status"] != "COMPLETED" or integrity["status"] != "PASS":
        raise RuntimeError("Completed accounting and integrity PASS required")
    if out == run or run in out.parents or out in run.parents:
        raise ValueError("Export must be separate from the raw evidence tree")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    text_extensions = {".json", ".jsonl", ".txt", ".log", ".md", ".yaml", ".yml", ".toml"}
    for source in sorted(x for x in run.rglob("*") if x.is_file()):
        relative = source.relative_to(run)
        size, checksum = source.stat().st_size, sha(source)
        copied = source.suffix.lower() in text_extensions or (source.suffix.lower() == ".csv" and size <= 2 * 1024 * 1024)
        destination = out / "snapshot" / relative
        if copied:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha(destination) != checksum:
                raise ValueError("Copy mismatch: " + str(relative))
        rows.append(dict(path=relative.as_posix(), size_bytes=size, sha256=checksum,
                         copied=copied, archive_path=("snapshot/" + relative.as_posix()) if copied else None,
                         omitted_reason=None if copied else "bulk trace/image retained at raw_root; no deletion"))
    (out / "results").mkdir()
    for source in sorted(a.results.iterdir()):
        if source.is_file():
            shutil.copyfile(source, out / "results" / source.name)
    shutil.copyfile(a.integrity, out / "final_integrity.json")
    plan = read(run / "plan.json")
    inventory = dict(schema="v64_c4a_complete_raw_inventory_v1", raw_root=str(run),
                     all_files=rows, total_files=len(rows), total_bytes=sum(r["size_bytes"] for r in rows),
                     copied_files=sum(r["copied"] for r in rows),
                     copied_bytes=sum(r["size_bytes"] for r in rows if r["copied"]))
    write(out / "raw_inventory_sha256.json", inventory)
    with (out / "README.md").open("x", encoding="utf8", newline="\n") as f:
        f.write("# C4-A final DEV evidence\n\n")
        f.write("Completed two-task DEV, not C.3 TEST. All search selections were sealed before Actual.\n\n")
        f.write("`snapshot/` contains unchanged configuration, tasks, noises, budgets, proposals, candidate registries, selections, commands, gate reports and seals. `results/` contains complete accounting; `final_integrity.json` records final frozen-byte and contract verification.\n\n")
        f.write("This is a compact review archive, not a self-contained raw-trace archive. Large trajectory CSVs, NPZs and generated plots remain at `" + str(run) + "`. Nothing was deleted. Every original file has its size, SHA-256 and copy status in `raw_inventory_sha256.json`. Embedded producer paths are preserved, not rewritten. For independent replay/trace verification, use that raw directory. No new physical execution is needed to inspect the evidence.\n\n")
        f.write("The `snapshot/plan.json` and split manifest also register both DEV mother seeds with the existing unused-seed inventory. Training updates: 0. C.3 TEST reruns: 0. New videos: none.\n")
    files = {x.relative_to(out).as_posix(): sha(x) for x in sorted(out.rglob("*")) if x.is_file()}
    write(out / "manifest.json", dict(schema="v64_c4a_review_release_v1", created_utc=datetime.now(timezone.utc).isoformat(),
          base_commit=plan["base_commit"], producer_commit=plan["producer_commit"], phase="DEV",
          physical_reruns_during_export=0, raw_root=str(run), self_contained_raw_traces=False,
          files_sha256=files, budget=summary["budget"]))
    for relative, expected in files.items():
        if sha(out / relative) != expected:
            raise ValueError("Export verification failed: " + relative)
    print(json.dumps(dict(status="PASS", output=str(out), files=len(files),
          total_raw_files=inventory["total_files"], total_raw_bytes=inventory["total_bytes"],
          copied_raw_files=inventory["copied_files"], copied_raw_bytes=inventory["copied_bytes"],
          manifest_sha256=sha(out / "manifest.json"))))


if __name__ == "__main__":
    main()
