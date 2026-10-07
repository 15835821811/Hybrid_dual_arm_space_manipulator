"""Package, authenticate, and safely import complete V6 evidence bundles.

The zip and its sidecar are release assets; raw traces need not enter Git.
Every included file has a SHA-256 entry, and extraction refuses path escape,
missing entries, altered bytes, and any existing destination directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_NAME = "bundle_manifest.json"


def _sha_stream(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _sha_path(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        return _sha_stream(stream)


def _safe_name(value: str) -> str:
    if "\\" in value or ":" in value or value.startswith("/"):
        raise ValueError(f"unsafe bundle path: {value!r}")
    parts = PurePosixPath(value).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe bundle path: {value!r}")
    normalized = PurePosixPath(*parts).as_posix()
    if normalized != value:
        raise ValueError(f"noncanonical bundle path: {value!r}")
    return normalized


def pack_evidence(roots: dict[str, Path], bundle_path: Path) -> dict[str, Any]:
    """Create a new archive and adjacent SHA-256 sidecar, never overwriting."""
    bundle_path = Path(bundle_path).resolve()
    sidecar = Path(str(bundle_path) + ".sha256.json")
    if bundle_path.exists() or sidecar.exists():
        raise FileExistsError(bundle_path if bundle_path.exists() else sidecar)
    if not roots:
        raise ValueError("at least one evidence root is required")
    files: list[tuple[str, Path]] = []
    for label, raw_root in roots.items():
        if _safe_name(label) != label or len(PurePosixPath(label).parts) != 1:
            raise ValueError(f"bundle root label must be one safe component: {label!r}")
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        if bundle_path.is_relative_to(root):
            raise ValueError("bundle output cannot lie inside an input root")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlink is not packageable: {path}")
            if path.is_file():
                name = _safe_name(f"{label}/{path.relative_to(root).as_posix()}")
                files.append((name, path))
    names = [name for name, _ in files]
    if len(names) != len(set(names)):
        raise ValueError("duplicate bundle member name")
    entries = []
    for name, path in files:
        digest, size = _sha_path(path)
        entries.append({"path": name, "sha256": digest, "bytes": size})
    manifest = {
        "schema": "v6_2_b1_evidence_bundle_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "file_count": len(entries),
        "total_uncompressed_bytes": sum(item["bytes"] for item in entries),
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, mode="x", allowZip64=True) as archive:
        for name, path in files:
            compression = (zipfile.ZIP_STORED if path.suffix.lower() == ".npz"
                           else zipfile.ZIP_DEFLATED)
            archive.write(path, arcname=name, compress_type=compression)
        archive.writestr(
            MANIFEST_NAME,
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
            compress_type=zipfile.ZIP_DEFLATED,
        )
    bundle_sha, bundle_bytes = _sha_path(bundle_path)
    sidecar_payload = {
        "schema": "v6_2_b1_evidence_bundle_sidecar_v1",
        "bundle_file": bundle_path.name,
        "bundle_sha256": bundle_sha,
        "bundle_bytes": bundle_bytes,
        "file_count": len(entries),
    }
    with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(sidecar_payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return sidecar_payload


def verify_bundle(bundle_path: Path, *, sidecar_path: Path | None = None,
                  import_dir: Path | None = None) -> dict[str, Any]:
    """Verify every byte and optionally import into a fresh directory."""
    bundle_path = Path(bundle_path)
    sidecar = Path(str(bundle_path) + ".sha256.json") if sidecar_path is None else Path(sidecar_path)
    if not bundle_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("bundle and SHA-256 sidecar are both required")
    expected = json.loads(sidecar.read_text(encoding="utf-8"))
    if expected.get("bundle_file") != bundle_path.name:
        raise ValueError("sidecar names a different bundle")
    actual_sha, actual_bytes = _sha_path(bundle_path)
    if (expected["bundle_sha256"] != actual_sha
            or expected["bundle_bytes"] != actual_bytes):
        raise ValueError("bundle hash or size mismatch")
    with zipfile.ZipFile(bundle_path, "r") as archive:
        raw_manifest = archive.read(MANIFEST_NAME)
        manifest = json.loads(raw_manifest)
        if manifest.get("schema") != "v6_2_b1_evidence_bundle_v1":
            raise ValueError("unsupported bundle manifest")
        entries = manifest["entries"]
        names = [_safe_name(item["path"]) for item in entries]
        if len(names) != len(set(names)):
            raise ValueError("duplicate manifest entry")
        archive_names = [_safe_name(item.filename) for item in archive.infolist()
                         if not item.is_dir()]
        if len(archive_names) != len(set(archive_names)):
            raise ValueError("duplicate archive member")
        if set(archive_names) != set(names) | {MANIFEST_NAME}:
            raise ValueError("archive entries differ from manifest")
        if manifest["file_count"] != len(entries):
            raise ValueError("bundle file count mismatch")
        if manifest["total_uncompressed_bytes"] != sum(item["bytes"] for item in entries):
            raise ValueError("bundle byte count mismatch")
        for item in entries:
            with archive.open(item["path"]) as stream:
                digest, size = _sha_stream(stream)
            if digest != item["sha256"] or size != item["bytes"]:
                raise ValueError(f"bundle member mismatch: {item['path']}")
        if import_dir is not None:
            import_dir = Path(import_dir)
            import_dir.mkdir(parents=True, exist_ok=False)
            root = import_dir.resolve()
            for item in entries:
                destination = import_dir.joinpath(*PurePosixPath(item["path"]).parts)
                if not destination.resolve().is_relative_to(root):
                    raise ValueError(f"bundle path escaped import root: {item['path']}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item["path"]) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if _sha_path(destination) != (item["sha256"], item["bytes"]):
                    raise ValueError(f"imported member mismatch: {item['path']}")
    return {
        "passed": True,
        "bundle_sha256": actual_sha,
        "file_count": len(entries),
        "total_uncompressed_bytes": manifest["total_uncompressed_bytes"],
        "import_dir": str(import_dir) if import_dir is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    pack = command.add_parser("pack")
    pack.add_argument("--bundle", type=Path, required=True)
    pack.add_argument("--include", action="append", required=True,
                      help="label=directory; may be repeated")
    verify = command.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--sidecar", type=Path)
    verify.add_argument("--import-dir", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        roots = {}
        for entry in args.include:
            if "=" not in entry:
                parser.error("--include must be label=directory")
            label, directory = entry.split("=", 1)
            if label in roots:
                parser.error(f"duplicate evidence label: {label}")
            roots[label] = Path(directory)
        result = pack_evidence(roots, args.bundle)
    else:
        result = verify_bundle(args.bundle, sidecar_path=args.sidecar,
                               import_dir=args.import_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
