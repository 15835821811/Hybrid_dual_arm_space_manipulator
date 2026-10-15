import copy
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from ff_prior.contracts import *
from ff_prior.spline import Reference,decode,KNOTS,from_arrays
from ff_prior.storage import Ledger,RunLock,atomic_json,read_json,digest
from ff_prior.dataset import Scaler,split_manifest,export_dataset


def condition():
    return {"initial_features":np.zeros(17),"goals":np.tile(np.eye(4),(2,1,1)),"duration":10.,"pose_mask":np.ones((2,6))}


def test_true_spline_endpoints_shared_terminal_and_no_extra_dwell():
    z = np.arange(19*17).reshape(19,17)/500
    ref = Reference(z,np.zeros(17),10)
    assert len(KNOTS)==28
    assert ref.control.shape==(24,17)
    for derivative in (1,2):
        np.testing.assert_allclose(ref.evaluate([0,10],derivative),0,atol=1e-12)
    assert np.linalg.norm(ref.evaluate(.01))>0
    np.testing.assert_allclose(ref.evaluate(10),z[-1],rtol=0,atol=2e-15)
    delta = np.zeros_like(z); delta[-1,0]=1e-6
    changed = decode(z+delta,np.zeros(17))-ref.control
    assert np.count_nonzero(changed)==3
    assert np.isclose(changed.sum(),3e-6)
    np.testing.assert_array_equal(from_arrays(ref.arrays()).control,ref.control)


def test_derivative_matches_numerical_and_time_scaling():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(19,17))*.01
    a,b = Reference(z,np.zeros(17),10),Reference(z,np.zeros(17),20)
    t = np.linspace(.1,9.9,30)
    h = 1e-5
    np.testing.assert_allclose(a.evaluate(t,1),(a.evaluate(t+h)-a.evaluate(t-h))/(2*h),atol=1e-8)
    np.testing.assert_allclose(a.evaluate(t,2),(a.evaluate(t+h,1)-a.evaluate(t-h,1))/(2*h),atol=1e-7)
    np.testing.assert_allclose(a.evaluate(t,1)/2,b.evaluate(t*2,1))
    np.testing.assert_allclose(a.evaluate(t,2)/4,b.evaluate(t*2,2))
    np.testing.assert_array_equal(a.evaluate(10.2,1),np.zeros(17))


@pytest.mark.parametrize("field",["q_terminal_seed","witness","terminal_base","actual_time","success_clearance"])
def test_future_condition_fields_rejected(field):
    c = condition(); c[field]=0
    with pytest.raises(ValueError): public_condition(c)


def test_frozen_goal_cannot_be_relabelled_after_failure():
    task = frozen_task(condition(),{"initial":"fixed"})
    assert_frozen(task)
    changed = copy.deepcopy(task)
    changed["condition"]["goals"][0,0,3] += .001
    with pytest.raises(ValueError): assert_frozen(changed)


@pytest.mark.parametrize("value",[None,False,"PASS",1])
def test_unknown_required_gate_cannot_pass(value):
    gates = dict.fromkeys(MODEL_GATES,True)
    gates["geometry"]=value
    assert not all_pass(gates,MODEL_GATES)


def pool(count=5,same_root=False,duplicates=False):
    return [{"sample_id":str(i),"root_id":"same" if same_root else str(i),"task_hash":"same_task",
        "validation_level":"MODEL_VALID","path":np.ones((128,17))*(0 if duplicates else i*.1)} for i in range(count)]


def test_strict_pool_four_copies_retimings_and_same_root_fail():
    for records in (pool(4),pool(5,duplicates=True),pool(5,same_root=True)):
        assert strict_pool(records,{r["sample_id"]:"EXECUTION_VERIFIED" for r in records})=="INSUFFICIENT_CANDIDATES"
    rows = pool()
    assert strict_pool(rows,{"0":"EXECUTION_VERIFIED"})=="INCOMPLETE_POOL_VERIFICATION"
    assert strict_pool(rows,{str(i):"EXECUTION_FAILED" for i in range(5)})=="COMPLETE"


def test_main_sample_admission_has_no_minfive():
    row = {"sample_id":"one","validation_level":"MODEL_VALID","execution_status":"NOT_RUN","data_role":"FF_MODEL_PRIOR"}
    manifest = {"records":{"one":{"split":"TRAIN"}}}
    assert export_dataset([row],manifest)["eligible"]["TRAIN"]==["one"]
    row["execution_status"]="FAIL"
    assert not export_dataset([row],manifest)["eligible"]["TRAIN"]


def test_train_only_scalers_constant_dimensions_and_schema(tmp_path):
    z = np.ones((3,19,17))*.02
    c = [condition() for _ in range(3)]
    with pytest.raises(ValueError): Scaler.fit(z,c,["TRAIN","VAL","TRAIN"],"robot","initial")
    scaler = Scaler.fit(z,c,["TRAIN"]*3,"robot","initial")
    assert np.isfinite(scaler.condition(c[0])).all()
    np.testing.assert_allclose(scaler.inverse_z(scaler.z(z)),z)
    scaler.save(tmp_path/"s.json")
    reloaded = Scaler.load(tmp_path/"s.json","robot","initial")
    np.testing.assert_array_equal(reloaded.condition(c[0]),scaler.condition(c[0]))
    with pytest.raises(ValueError): Scaler.load(tmp_path/"s.json","different","initial")
    a,b = condition(),condition()
    quat = np.array([.2,.3,.1,.9]); quat/=np.linalg.norm(quat)
    a["goals"][0,:3,:3] = Rotation.from_quat(quat).as_matrix()
    b["goals"][0,:3,:3] = Rotation.from_quat(-quat).as_matrix()
    np.testing.assert_allclose(public_condition(a),public_condition(b))


def test_goal_neighborhood_lineage_and_empty_scene_split():
    rows=[]
    for i in range(30):
        c=condition(); c["goals"][:,0,3]=i*.2
        rows.append({"sample_id":str(i),"family_id":str(i),"condition":c,"source_partition":"main","scene_family_id":None})
    rows.append(dict(rows[0],sample_id="revision"))
    rows.append(dict(rows[0],sample_id="neighbor",family_id="neighbor"))
    split=split_manifest(rows)
    assert split["records"]["0"]["split"]==split["records"]["revision"]["split"]==split["records"]["neighbor"]["split"]
    assert len({r["split"] for r in split["records"].values()})==3


def test_run_lock_is_exclusive(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(FileExistsError):
            with RunLock(tmp_path): pass
    assert not (tmp_path/"run.lock").exists()


def test_resume_attempts_duplicates_identity_and_interruption(tmp_path):
    ledger=Ledger(tmp_path,{"run":1})
    assert ledger.begin("a","main",100,2)
    ledger.finish({"status":"REJECTED"},{"distance":7})
    again=Ledger(tmp_path,{"run":1})
    assert not again.begin("a","main",100,2)
    assert again.data["calls"]["main.distance"]==7
    assert again.begin("b","main",100,2)
    crashed=Ledger(tmp_path,{"run":1})
    assert crashed.data["completed"]["b"]["status"]=="INTERRUPTED"
    with pytest.raises(RuntimeError,match="BUDGET_EXHAUSTED"): crashed.begin("c","main",100,2)
    with pytest.raises(ValueError,match="identity"): Ledger(tmp_path,{"run":2})
