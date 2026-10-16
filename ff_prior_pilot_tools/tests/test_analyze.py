import os
import shutil
from pathlib import Path
import numpy as np
import pytest
from ff_prior.storage import read_json,atomic_json,atomic_npz
from ff_prior_pilot_tools.analyze import analyze


def test_real_saved_array_checks_detect_corruption_and_missing_record(tmp_path):
    actual=Path('artifacts/v7_1/qualified_pilot_001');source=tmp_path/'fixture';output=tmp_path/'analysis'
    sid='pilot_train_00000';row=read_json(actual/'records'/f'{sid}.json')
    paths=[f'records/{sid}.json',row['arrays'],row['complete_initial_state'],row['generation_proof'],f'tasks/{sid}/task.json']
    for rel in paths:
        dest=source/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(actual/rel,dest)
    atomic_json(source/'status.json',{'pid':os.getpid(),'status':'RUNNING','stage':'unit_fixture'})
    atomic_json(source/'ledger.json',{'completed':{sid:{'status':'ADMITTED'}}})
    result=analyze(source,output)
    assert result['status']=='PASS_SAVED_ARRAY_AUDIT' and result['committed_saved_rows_checked']==1
    assert result['validation_levels']=={'MODEL_VALID_ONLY':1} and not result['full_run_success_claim']
    with np.load(source/row['arrays']) as a:values={k:a[k].copy() for k in a.files}
    values['view_q'][0,0]+=.01;atomic_npz(source/row['arrays'],**values)
    assert analyze(source,output)['status']=='FAILED_SAVED_ARRAY_AUDIT'
    (source/'records'/f'{sid}.json').unlink()
    assert analyze(source,output)['issues']==[{'sample_id':sid,'error':'MISSING_COMMITTED_RECORD'}]


def test_output_cannot_mutate_original_run(tmp_path):
    with pytest.raises(ValueError,match='separate'):
        analyze(tmp_path,tmp_path/'analysis')
