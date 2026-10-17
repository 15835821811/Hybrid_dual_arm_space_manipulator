"""Real scene eviction/rebuild must preserve physical identity and propagation."""
import gc
import numpy as np
import pytest
from ff_prior.storage import read_json,digest
from ff_prior.pipeline import audit,acceptance
from ff_prior.guidance import public_goal_initializer
from ff_prior.coupled_predictor import predict_coupled
from ff_prior.spline import from_arrays
from ff_prior_origin_scene.runtime import at_scene_origin
from ff_prior_origin_learn_bounded.runtime import TaskOriginSceneCache,memory_usage
from ff_prior_origin_learn_bounded.protocol import ROOT,SOURCE,BASE


def test_real_scene_eviction_preserves_state_initializer_and_coupled_prediction():
    config=read_json(BASE/'run.json')['config'];_,rr,ii,_=audit(config);profile=acceptance(config)
    entries=read_json(SOURCE/'training_index.json')['records'];rows=[]
    for e in entries:
        if not e['sample_id'].startswith('batch_v1_train_'):continue
        r=read_json(ROOT/e['record'])
        if r.get('scene',{}).get('objects'):rows.append((e,r))
        if len(rows)==6:break
    assert len(rows)==6
    cache=TaskOriginSceneCache(rr,ii,profile,max_entries=2,max_scene_models=1)
    e,row=rows[0];a,ai,ac,ap=cache.get(row)
    with np.load(ROOT/e['arrays']) as archive:ref=from_arrays(archive)
    base=public_goal_initializer(row['task'],a,ai)
    pred=predict_coupled(ai,ref,a,32)
    original_values=ai.values.copy();original_model=a.identity()['scene_model_hash'];original_proof=digest(ap)
    del a,ai,ac,ap;gc.collect()
    for _,entry in rows[1:]:cache.get(entry)
    a,ai,ac,ap=cache.get(row)
    assert np.array_equal(ai.values,original_values) and a.identity()['scene_model_hash']==original_model
    assert digest(ap)==original_proof and np.array_equal(public_goal_initializer(row['task'],a,ai),base)
    again=predict_coupled(ai,ref,a,32)
    for key in ('qpos','base','ee','body','geom','geom_rotation','velocity'):assert np.array_equal(pred[key],again[key])
    stat=cache.stats();assert stat['evictions']==6 and stat['builds']==7
    assert stat['peak_entries']<=2 and stat['peak_scene_models']==1
    assert stat['peak_private_bytes']<16*1024**3


def test_actual_windows_memory_measurement_and_guard():
    m=memory_usage()
    assert 0<m['private_bytes'] and 0<m['working_bytes']
    assert 0<m['available_physical_bytes']<=m['total_physical_bytes']
    cache=TaskOriginSceneCache(None,None,None,max_private_bytes=0,min_available_bytes=0)
    with pytest.raises(RuntimeError,match='RUNTIME_MEMORY_BUDGET_EXHAUSTED'):cache.guard()
