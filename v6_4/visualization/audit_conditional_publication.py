"""Read-only B.3 publication QA: static artifacts and saved-file fidelity.

No repository module is imported. No browser, HTTP server, physics, optimizer,
sampler, geometry backend, or dynamic JavaScript is started. --output accepts
only a new file outside the repository; stdout is the default destination.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path("E:/v64b3work_20261007_01")
RUN = "conditional_route_value_20261007_01"
METHODS = ("Z0", "R0", "U0", "D_true", "D_swap")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def close(actual, expected, atol=1e-12):
    return math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=0., abs_tol=atol)


def subset_equal(actual, expected):
    if isinstance(actual, dict):
        return isinstance(expected, dict) and all(key in expected and subset_equal(value, expected[key])
            for key, value in actual.items())
    return actual == expected


class Html(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.ids, self.scripts, self.tables, self.text = [], set(), [], {}, []
        self.section = None
        self.row = None
        self.cell = None
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "section":
            self.section = attrs.get("id")
        if tag == "script":
            self.scripts.append(attrs)
        for key in ("src", "href"):
            if key in attrs:
                self.links.append({"tag": tag, "attribute": key, "value": attrs[key]})
        if tag == "tr":
            self.row = []
        if tag in ("td", "th") and self.row is not None:
            self.cell = []
    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.tables.setdefault(self.section, []).append(self.row)
            self.row = None
        if tag == "section":
            self.section = None
    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None:
            self.cell.append(data)


class Audit:
    def __init__(self, root, final=False):
        self.root = Path(root).resolve()
        self.out = self.root/"v6_4/output"/RUN
        self.vis = self.root/"v6_4/visualization"/RUN
        self.release = self.root/"v6_4/releases"/RUN
        self.final = final
        self.errors, self.pending, self.checks, self.hashes = [], [], [], {}
    def require(self, condition, description):
        self.checks.append(description)
        if not condition:
            self.errors.append(description)
    def checksum(self, path, row):
        path = Path(path).resolve()
        if not path.is_file():
            self.require(False, "missing artifact: "+str(path))
            return
        if path not in self.hashes:
            self.hashes[path] = sha(path)
        self.require(self.hashes[path] == row["sha256"] and path.stat().st_size == row["bytes"],
            "SHA256/size: "+str(path))
    def manifest(self, name):
        manifest = read(self.vis/name)
        for row in manifest.get("artifacts", []):
            self.checksum(self.vis/row["path"], row)
        for row in manifest.get("inputs", []):
            self.checksum(self.out/row["path"], row)
        for row in manifest.get("visualization_inputs", []):
            self.checksum(self.vis/row["path"], row)
        generator = manifest.get("generator", {})
        if generator:
            self.checksum(self.root/"v6_4/visualization"/generator.get("path", generator.get("name")), generator)
        for key in ("physics_steps", "DDIM_calls", "optimizer_updates", "geometry_queries"):
            self.require(manifest.get(key) == 0, name+": zero new "+key)
        return manifest
    def saved_data(self, plot, quality, paired):
        qualities = {row["slot_id"]: row for row in quality}
        slots = {row["slot_id"]: row for row in paired["pilot"]["slots"]}
        plotted = []
        for scene in plot["scenes"]:
            tid = scene["task_id"]
            declaration = read(self.out/"tasks"/tid/"task.json")
            canonical = json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False)
            self.require(hashlib.sha256(canonical.encode("utf8")).hexdigest() == scene["task_sha256"], tid+": declared task identity")
            self.require(scene["key_interval_slot"] == 2, tid+": fixed key interval")
            for record in scene["actual_records"]:
                slot = record["slot_id"]
                plotted.append(slot)
                q, p = qualities[slot], slots[slot]
                self.require(record["task_id"] == tid == q["task_id"] == p["task_id"], slot+": task binding")
                self.require(record["candidate_name"] == q["candidate_name"] == p["method"], slot+": candidate binding")
                self.require(subset_equal(record["full_metrics"], q["full_metrics"]), slot+": displayed quality fields exactly match original metrics")
                for key in ("I_route_rad_s", "I_full_rad_s", "continuum_path_length_m", "base_translation_peak_m", "base_rotation_peak_rad"):
                    self.require(close(record["full_metrics"][key], p[key]), slot+": plot/paired "+key)
                self.require(record["complete_quality_eligible"] is True and record["full_task_success"] is True,
                    slot+": complete safe quality eligibility")
                self.require(record["actual_physics_steps"] == 13500 and record["saved_horizon_s"] == 27., slot+": complete 27s source scope")
                source = self.out/record["state_source"]
                self.checksum(source, {"sha256": record["source_fresh_sha256"], "bytes": source.stat().st_size})
                with np.load(source, allow_pickle=False) as state:
                    times = state["time"]
                    positions = state["continuum_position"]
                    selected = np.asarray(record["selected_source_state_indices"], dtype=int)
                    self.require(len(times) == record["source_state_count"] == 13501, slot+": native saved state count")
                    self.require(selected.ndim == 1 and selected[0] == 0 and selected[-1] == len(times)-1
                        and np.all(np.diff(selected) > 0) and selected.min() >= 0 and selected.max() < len(times), slot+": source-only ordered endpoint-preserving thinning")
                    self.require(np.array_equal(times[selected], np.asarray(record["time_s"])), slot+": plotted times equal saved source rows")
                    self.require(np.array_equal(positions[selected], np.asarray(record["continuum_position_world_m"])), slot+": plotted positions equal saved source rows")
                    self.require(record["exact_saved_endpoint_included"] is True, slot+": exact final source endpoint")
                    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
                    self.require(close(path_length, p["continuum_path_length_m"]), slot+": full path length from saved states")
                attempt = read(self.out/"pilot/attempts"/slot/"attempt_result.json")
                with np.load(attempt["trace_path"], allow_pickle=False) as trace:
                    tick_count = attempt["actual_steps"]//10
                    t = trace["task_time"][:tick_count]
                    norms = trace["task_avoidance_intervention"][:tick_count]
                    lo, hi = scene["T_route_s"]
                    mask = (t >= lo-1e-10)&(t <= hi+1e-10)
                    self.require(close(float(np.sqrt(np.mean(norms[mask]**2))), p["I_route_rad_s"]), slot+": I_route from original closed-window producer norms")
                    self.require(close(float(np.sqrt(np.mean(norms**2))), p["I_full_rad_s"]), slot+": full intervention from producer norms")
                clearance = q["full_metrics"]["continuum_route_obstacle_clearance"]
                self.require(close(clearance["route_window_minimum_m"], p["route_clearance_m"])
                    and close(clearance["full_saved_horizon_minimum_m"], p["full_clearance_m"]), slot+": relevant continuum-sphere clearance mapping")
                self.require(all(p["safety"]["gates"].values()) and p["safety"]["historical_runtime_reported_passed"] is False,
                    slot+": original five gates retained; historical strict-curve failure not relabeled")
            definitions = read(self.out/"tasks"/tid/"definition.json")
            for candidate in scene["reference_candidates"]:
                mode = {"z0": "z0", "z+": "z_plus", "z-": "z_minus"}[candidate["mode"]]
                plan = read(self.out/"fixed_candidates"/tid/mode/"plan.json")
                self.require(plan["z_m"] == candidate["z_m"] and plan["definition"] == definitions, tid+": displayed frozen reference "+mode)
        self.require(sorted(plotted) == sorted(slots) and len(plotted) == 6, "exactly all six pilot slots plotted, no mirror/substitution")
    def tables(self, report, paired, html):
        with (self.vis/"method_table.csv").open(encoding="utf8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.require((self.out/"method_table.csv").read_bytes() == (self.vis/"method_table.csv").read_bytes(), "source/dashboard CSV byte parity")
        self.require(len(rows) == 6, "CSV includes six pilot rows")
        expected = paired["pilot"]["slots"]
        for actual, row in zip(rows, expected):
            for key, value in actual.items():
                if isinstance(row[key], bool):
                    passed = value == str(row[key])
                elif isinstance(row[key], (int, float)):
                    passed = close(value, row[key])
                else:
                    passed = value == row[key]
                self.require(passed, row["slot_id"]+": CSV "+key)
        html_rows = [row for row in html.tables.get("pilot", []) if row and "PILOT_" in row[0]]
        self.require(len(html_rows) == 6, "HTML pilot table includes six rows")
        numeric_columns = ((3, "saved_horizon_s", 3, 1.), (4, "I_route_rad_s", 6, 1.),
            (5, "I_full_rad_s", 6, 1.), (6, "continuum_path_length_m", 4, 1.),
            (7, "route_clearance_m", 3, 1000.), (8, "full_clearance_m", 3, 1000.))
        for actual, row in zip(html_rows, expected):
            self.require(row["slot_id"] in actual[0] and row["task_id"] in actual[0] and actual[1] == row["method"], row["slot_id"]+": HTML row identity")
            for column, key, digits, factor in numeric_columns:
                self.require(actual[column] == f"{row[key]*factor:.{digits}f}", row["slot_id"]+": HTML formatted "+key)
        markdown = (self.out/"REPORT.md").read_text(encoding="utf8")
        md_rows = [line for line in markdown.splitlines() if line.startswith("| PILOT_")]
        self.require(len(md_rows) == 6, "REPORT.md includes all six actual records")
        for line, row in zip(md_rows, expected):
            columns = [part.strip() for part in line.split("|")[1:-1]]
            self.require(columns[:4] == [row["slot_id"], row["task_id"], row["method"], row["status"]], row["slot_id"]+": report row identity")
            self.require(close(columns[6], row["I_route_rad_s"], 5e-9) and close(columns[7], row["route_clearance_m"]*1000., 5e-7), row["slot_id"]+": report rounded quality numbers")
        self.require(report["decision"] == paired["pilot"]["decision"], "report/paired decision exact parity")
        self.require(report["costs"] == paired["costs"], "report/paired cost exact parity")
        self.require(report["route_value_identifiable"] is False and report["training_executed"] is False,
            "negative pilot stop and no training preserved")
        self.require(set(report["independent_test_task_success_by_method"]) == set(METHODS)
            and all(value == "NOT_RUN_PILOT_STOP" for value in report["independent_test_task_success_by_method"].values()),
            "five formal TEST methods explicitly unexecuted")
        if self.final:
            self.require(report["research_delivery_complete"] is True, "final reviewed research delivery complete")
        elif report["research_delivery_complete"] is not True:
            self.pending.append("REVIEW_DELIVERY_FLAG_PENDING")
        self.require(report["deployment"] == "NOT_MET" and report["20ms_wall_is_research_gate"] is False,
            "deployment and 20ms research-scope labels preserved")
    def links(self, html_path):
        value = html_path.read_text(encoding="utf8")
        parsed = Html(); parsed.feed(value)
        for link in parsed.links:
            url = urlsplit(link["value"])
            if url.scheme or url.netloc:
                self.require(link["tag"] != "script" and link["attribute"] != "src"
                    and url.scheme in ("http", "https", "mailto"), "no external executable or resource URL: "+link["value"])
                continue
            if not url.path:
                self.require(not url.fragment or unquote(url.fragment) in parsed.ids, "HTML internal anchor exists: "+link["value"])
                continue
            path = (html_path.parent/unquote(url.path)).resolve()
            self.require(path.is_relative_to(self.root), "HTML local link stays within repository: "+link["value"])
            if not path.exists() and path.is_relative_to(self.release) and not self.final:
                self.pending.append("PENDING_EXPORT: "+link["value"])
            else:
                self.require(path.exists(), "HTML local src/href exists: "+link["value"])
            if path.suffix == ".html" and path.is_file() and url.fragment:
                child = Html();child.feed(path.read_text(encoding="utf8"))
                self.require(unquote(url.fragment) in child.ids, "linked HTML fragment exists: "+link["value"])
        self.require(not parsed.scripts, "dashboard has no dynamic or external JavaScript")
        self.require(not re.search(r"@import\s|url\s*\(\s*['\"]?(?:https?:|//)", value, re.I), "dashboard CSS has no external resources")
        return parsed
    def entries(self):
        for relative in ("README.md", "V6_lite/README.md", "docs/V6_4_B3_VISUALIZATION.md"):
            source = self.root/relative
            content = source.read_text(encoding="utf8")
            self.require(RUN+"/index.html" in content, relative+": current dashboard entry")
            for target in re.findall(r"\]\(([^)]+)\)", content):
                if RUN not in target and "V6_4_B3_VISUALIZATION.md" not in target:
                    continue
                path = (source.parent/unquote(urlsplit(target).path)).resolve()
                self.require(path.is_relative_to(self.root), relative+": current link stays within repository: "+target)
                if not path.exists() and path.is_relative_to(self.release) and not self.final:
                    self.pending.append("PENDING_EXPORT: "+target)
                else:
                    self.require(path.is_file(), relative+": current link exists: "+target)
        redirect = self.root/"V6_lite/visualization/index.html"
        self.links(redirect)
        content = redirect.read_text(encoding="utf8")
        match = re.search(r'http-equiv="refresh"\s+content="0;url=([^"]+)"', content, re.I)
        self.require(bool(match) and (redirect.parent/match.group(1)).resolve() == self.vis/"index.html", "legacy visualization entry redirects to current B.3 dashboard")
    def run(self):
        figure_manifest = self.manifest("manifest.json")
        self.manifest("dashboard_manifest.json")
        plot = read(self.vis/"plot_data.json")
        report, paired = read(self.out/"report.json"), read(self.out/"paired_metrics.json")
        quality = read(self.out/"pilot/quality_records.json")
        self.require(plot["input_sources"] == figure_manifest["inputs"], "plot-data input ledger matches figure manifest exactly")
        self.require(plot["pilot_decision"] == report["decision"] == read(self.out/"pilot/decision.json"), "plot/report/source pilot decision exact parity")
        self.require(len(plot["panels"]) == len(plot["scenes"]) == 2, "exactly two paired route panels")
        for panel in plot["panels"]:
            with Image.open(self.vis/panel["png"]) as picture:
                image = np.asarray(picture.convert("RGB"))
                self.require(picture.width >= 2000 and picture.height >= 1400 and float(image.std()) > 5., panel["id"]+": readable nonblank PNG dimensions")
            pdf = (self.vis/panel["pdf"]).read_bytes()
            self.require(pdf.startswith(b"%PDF-") and b"%%EOF" in pdf[-1024:], panel["id"]+": complete PDF file structure")
        self.saved_data(plot, quality, paired)
        html = self.links(self.vis/"index.html")
        self.tables(report, paired, html)
        self.entries()
        if not self.release.exists():
            if self.final:
                self.require(False, "final portable release exists")
            else:
                self.pending.append("PENDING_EXPORT")
        else:
            release_manifest = read(self.release/"release_manifest.json")
            release_verification = read(self.release/"release_verification.json")
            self.require(release_verification["status"] == "PASS"
                and release_verification["release_manifest_sha256"] == sha(self.release/"release_manifest.json"), "portable release verification binding")
            for row in release_manifest["included"].values():
                self.checksum(self.release/row["release_relative_path"], row)
            self.require((self.release/"snapshot/report.json").read_bytes() == (self.out/"report.json").read_bytes(), "portable original report byte parity")
        return {"schema": "v64_b3_static_publication_audit_v1", "status": "FAIL" if self.errors else "PASS",
            "mode": "FINAL" if self.final else "PRE_EXPORT", "checked_conditions": len(self.checks),
            "verified_unique_file_hashes": len(self.hashes), "errors": self.errors, "pending": sorted(set(self.pending)),
            "scope": "pure saved-file/static HTML/CSV/data QA; no browser interaction or runtime rendering validation",
            "browser_automation": "NOT_VERIFIED_TOOL_POLICY", "visual_model_review": "separate model view_image inspection",
            "physics_steps": 0, "geometry_queries": 0, "DDIM_calls": 0, "optimizer_updates": 0,
            "auditor_sha256": sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = Audit(args.root, args.final)
    try:
        result = audit.run()
    except Exception as error:
        result = {"schema": "v64_b3_static_publication_audit_v1", "status": "FAIL",
            "errors": [*audit.errors, f"{type(error).__name__}: {error}"], "pending": audit.pending,
            "physics_steps": 0, "geometry_queries": 0, "DDIM_calls": 0, "optimizer_updates": 0}
    if args.output:
        output = args.output.resolve()
        if output.is_relative_to(audit.root):
            parser.error("--output must be outside the repository and study/visualization outputs")
        with output.open("x", encoding="utf8", newline="\n") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
