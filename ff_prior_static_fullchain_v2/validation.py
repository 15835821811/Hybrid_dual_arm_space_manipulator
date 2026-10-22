"""Extra validation density, unchanged acceptance limits and model semantics."""
import numpy as np
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from ff_prior.coupled_predictor import predict_coupled, pose_errors
from ff_prior.model_gates import validate_model
from ff_prior_feasible.validation import compare
from ff_prior_pilot_train.storage import atomic_json, atomic_npz
from .protocol import P

class RecordedCollision:
    """Record original validator scans; does not add or alter geometry queries."""
    def __init__(self,delegate):
        self.delegate=delegate;self.states=[];self.records=[]
    def __getattr__(self,name):return getattr(self.delegate,name)
    def scan(self,data,threshold):
        r=self.delegate.scan(data,threshold)
        self.states.append(data.qpos.copy())
        self.records.append({'minimum_clearance_m':r['minimum_clearance_m'],
                             'minimum_pair':r['minimum_pair'],'threshold_m':threshold})
        return r

def validate(directory, z, task, robot, initial, collision, profile, budget):
    c = task["condition"]
    ref = Reference(z, c["initial_features"], c["duration"])
    quick = quick_gates(robot, ref)
    r = {"status": "DONE", "eligible": False, "frozen_goal_pass": False, "Z": z,
         "reference_hash": ref.reference_hash, "task_hash": task["task_hash"],
         "validation_grids": [], "goal_relabel": False, "training_eligible": False}
    if not all(quick.values()):
        r["validation"] = {"validation_level": "REJECTED", "gates": quick}
    else:
        budget.consume(256)
        coarse = predict_coupled(initial, ref, robot, 256, budget.deadline)
        atomic_npz(directory/'grid_256.npz',**{k:v for k,v in coarse.items() if k!='numerical'})
        atomic_json(directory/'grid_256_numerical.json',coarse['numerical'])
        comparisons, penetrated = [], False
        for n in P["validation_intervals"][1:]:
            budget.consume(n)
            fine = predict_coupled(initial, ref, robot, n, budget.deadline)
            atomic_npz(directory/f'grid_{n}.npz',**{k:v for k,v in fine.items() if k!='numerical'})
            atomic_json(directory/f'grid_{n}_numerical.json',fine['numerical'])
            comparisons.append(compare(coarse, fine, robot))
            recorded=RecordedCollision(collision)
            gates = validate_model(ref, fine, profile, robot, initial, recorded,
                {"passed": comparisons[-1]["passed"], "comparisons": comparisons.copy()}, budget.deadline)
            atomic_npz(directory/f'grid_{n}_geometry_scans.npz',qpos=np.asarray(recorded.states),
                       minima=np.asarray([v['minimum_clearance_m'] for v in recorded.records]))
            atomic_json(directory/f'grid_{n}_geometry_scans.json',recorded.records)
            r["validation_grids"].append({"intervals": n, "gates": gates["gates"].copy(),
                                          "geometry": gates["geometry"]})
            penetrated = penetrated or not gates["gates"]["geometry"]
            if penetrated:
                gates["gates"]["geometry"] = False
                gates["validation_level"] = "REJECTED"
            if gates["gates"]["geometry_refinement"] and gates["gates"]["coupled_convergence"]:
                break
            coarse = fine
        pos, rot = pose_errors(fine["ee"][-1], c["goals"])
        r["frozen_goal_pass"] = bool(np.all(pos <= profile.position_error_m) and np.all(rot <= profile.rotation_error_rad))
        gates.update(frozen_goal_position_error_m=pos, frozen_goal_rotation_error_rad=rot)
        r.update(validation=gates, eligible=r["frozen_goal_pass"] and gates["validation_level"] == "MODEL_VALID",
                 score=float(np.sum((pos/profile.position_error_m)**2)+np.sum((rot/profile.rotation_error_rad)**2)))
        atomic_npz(directory / "prediction.npz", qpos=fine["qpos"], ee=fine["ee"], base=fine["base"],
                   phase=fine["phase"], geom=fine["geom"], geom_rotation=fine["geom_rotation"])
    atomic_json(directory / "validation.json", r)
    return r
