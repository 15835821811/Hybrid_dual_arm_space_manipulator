"""Actual retained failure/reservation and real Monitor; no fake physics passes."""
import copy
import shutil
import pytest
from ff_prior.storage import read_json,digest
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_coverage_train.job import Monitor
from ff_prior_generalization_recover.experiment import SOURCE,ROOT,inspect_source,known_zero_failure,diagnostic_progress,DiagnosticLedger


@pytest.fixture(scope='module')
def retained():
    run,manifest,proof=inspect_source(SOURCE)
    return run,manifest,proof


def test_real_monitor_keyword_binding_and_fixed_callback(tmp_path):
    with Monitor(tmp_path) as monitor:
        with pytest.raises(TypeError,match='multiple values'):
            monitor.update('independent_pre_guidance_validation',stage='raw')
        diagnostic_progress(monitor.update,'newval_00','structured','raw')
        s=read_json(tmp_path/'status.json')
        assert s['stage']=='independent_pre_guidance_validation' and s['progress']['proposal_stage']=='raw'


def test_actual_pre_physics_failure_and_tamper_refusal(retained):
    run,_,proof=retained
    assert proof['status']=='PROVEN_ZERO_DIAGNOSTIC_PHYSICS' and proof['failed_diagnostic_coupled_calls']==0
    ledger=read_json(SOURCE/'ledger.json');failure=read_json(SOURCE/'failure.json');state=read_json(SOURCE/'status.json')
    code=(ROOT/'ff_prior_generalization/experiment.py').read_text(encoding='utf-8-sig')
    bad=copy.deepcopy(ledger);bad['calls']['diagnostic.coupled']=1
    with pytest.raises(ValueError):known_zero_failure(run,state,failure,bad,code)
    bad=copy.deepcopy(failure);bad['reason']='unrelated interruption'
    with pytest.raises(ValueError):known_zero_failure(run,state,bad,ledger,code)


def test_same_reservation_count_is_retained(retained,tmp_path):
    run,_,proof=retained;shutil.copy2(SOURCE/'ledger.json',tmp_path/'ledger.json')
    ledger=DiagnosticLedger(tmp_path,{'identity_hash':digest(run['identity'])},proof['original_reservation'])
    assert ledger.data['attempts']['diagnostic']==1 and ledger.reused_reservation==proof['original_reservation']['id']
    # Bookkeeping-only unit result; never represents a physical validation.
    ledger.finish({'status':'SOFTWARE_ONLY_NO_PHYSICS'},{})
    ledger.begin('unit_next_reservation','diagnostic',600,96)
    assert ledger.data['attempts']['diagnostic']==2


def test_uncertain_started_diagnostic_is_not_repeated(retained,tmp_path):
    run,_,proof=retained;shutil.copy2(SOURCE/'ledger.json',tmp_path/'ledger.json')
    atomic_json(tmp_path/'diagnostics_started.json',{'already_started':True})
    with pytest.raises(RuntimeError,match='not repeated'):
        DiagnosticLedger(tmp_path,{'identity_hash':digest(run['identity'])},proof['original_reservation'])
    assert read_json(tmp_path/'ledger.json')['unresolved_call_accounting']
