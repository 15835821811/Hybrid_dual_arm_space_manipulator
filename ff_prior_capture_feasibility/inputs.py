import ast
import numpy as np
from ff_prior.storage import read_json, file_digest, digest
from ff_prior_pilot_train.storage import atomic_json
from .protocol import ROOT, SOURCE, BASE, PACKAGE, P

def sources():
    pending = list(PACKAGE.glob("*.py")); found = set()
    def add(module):
        parts = module.split(".")
        for i in range(1, len(parts)+1):
            p = ROOT.joinpath(*parts[:i])
            for path in (p.with_suffix(".py"), p/"__init__.py"):
                if path.is_file() and path not in found:
                    pending.append(path)
    while pending:
        p = pending.pop()
        if p in found:
            continue
        found.add(p)
        pkg = p.relative_to(ROOT).with_suffix("").parts[:-1]
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                for name in node.names: add(name.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(pkg[:len(pkg)-node.level+1]) if node.level else []
                module = ".".join(prefix+([node.module] if node.module else []))
                add(module)
                for name in node.names:
                    if name.name != "*": add(module+"."+name.name)
    found.add(PACKAGE/"job.ps1")
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(found)}

def freeze():
    destination = PACKAGE/"evidence/selection.json"
    if destination.exists(): raise ValueError("freeze is once-only")
    files = {}
    def get(p):
        files[p.relative_to(ROOT).as_posix()] = file_digest(p)
        return read_json(p)
    original = get(SOURCE/"frozen_tasks.json")["application"]["entries"]
    target = next(e for e in original if e["sample_id"] == P["task"])
    assert target["initial_status"] == "PASS" and len(target["scene"]["objects"]) == 5
    seed0 = get(SOURCE/"application/capture_5_obstacles/aligned/candidates/00.json")
    seeds = [{"id": "00_original_failed", "Z": seed0["Z"], "source_qualified_for_five": False}]
    for method in ("aligned", "plain", "structured"):
        root = SOURCE/"application/capture_3_obstacles"/method
        selected = get(root/"execution/selection.json")["selected_id"]
        r = get(root/"candidates"/f"{selected:02d}.json")
        assert r["eligible"]
        seeds.append({"id": f"{len(seeds):02d}_three_{method}", "Z": r["Z"],
                      "source_qualified_for_five": False})
    phase = np.r_[np.linspace(0, 1, 20)[1:-1], 1.]
    bow = np.sin(np.pi*phase)**2
    bow[-1] = 0.
    for j in P["seed_joints"]:
        for sign in (-1, 1):
            z = np.asarray(seed0["Z"]).copy()
            z[:, j] += sign*P["seed_offset_rad"]*bow
            seeds.append({"id": f"{len(seeds):02d}_bow_j{j}_{sign}", "Z": z,
                          "joint": j, "sign": sign, "source_qualified_for_five": False})
    assert len(seeds) == P["seeds"]
    for p in (SOURCE/"run.json", SOURCE/"independent_verification.json", SOURCE/"physical_contract.json",
              SOURCE/"learning/aligned/codec.json", SOURCE/"learning/aligned/model.pt",
              SOURCE/"learning/aligned/training.json", BASE/"run.json"):
        files[p.relative_to(ROOT).as_posix()] = file_digest(p)
    old_goal = ROOT/"DIFFUSION_GOAL_CURRENT.md"
    (PACKAGE/"evidence").mkdir(parents=True, exist_ok=True)
    (PACKAGE/"evidence/previous_goal.md").write_bytes(old_goal.read_bytes())
    selection = {"entry": target, "seeds": seeds, "task_hash": target["task"]["task_hash"],
                 "before_any_new_physics": True, "seed_source_only_not_five_scene_witness": True}
    atomic_json(PACKAGE/"evidence/binding.json", {"files": files,
        "previous_source_hashes": read_json(SOURCE/"run.json")["identity"]["source_hashes"]})
    atomic_json(destination, selection)
    return selection

def identity():
    b = read_json(PACKAGE/"evidence/binding.json")
    for p, h in dict(b["files"], **b["previous_source_hashes"]).items():
        if file_digest(ROOT/p) != h: raise ValueError("changed frozen input: "+p)
    s = read_json(PACKAGE/"evidence/selection.json")
    assert s["task_hash"] == s["entry"]["task"]["task_hash"]
    return {"protocol": P, "source_hashes": sources(), "binding": b, "selection": s}
