"""Algebra/identity tests only, never physics qualification."""
import ast
import numpy as np
import pytest
from ff_prior.storage import read_json
from ff_prior_goal.projection import smooth
from ff_prior_pilot_train.storage import atomic_json
from .protocol import SOURCE,ROOT,P
from .representation import fixed_basis,project,prepare
from .experiment import assert_resume_safe

def basis():return np.asarray(read_json(SOURCE/'learning/aligned/codec.json')['basis'])

@pytest.mark.parametrize('rank',[6,10,14,19])
def test_basis_is_orthonormal_and_contains_old_train_subspace(rank):
    old=basis();b=fixed_basis(old,rank)
    assert np.max(abs(b.T@b-np.eye(rank)))<1e-12
    assert np.max(abs(b@(b.T@old)-old))<1e-12
    if rank>6:assert np.max(abs(b@(b.T@np.eye(19)[-1])-np.eye(19)[-1]))<1e-12

def test_full_rank_preserves_exact_controls_and_hash_input():
    z=np.arange(323).reshape(19,17)/1000;bb=np.zeros_like(z)
    assert np.array_equal(project(z,bb,fixed_basis(basis(),19)),z)

def test_preserve_prepare_is_identity_for_valid_input():
    z=np.arange(323).reshape(19,17)/1000
    out,meta=prepare(z,np.zeros_like(z),np.eye(19),np.zeros(17),-np.ones(17),np.ones(17),lambda x:True,'preserve_prepare')
    assert np.array_equal(out,z) and meta['unchanged']

def test_legacy_smooths_actual_representation_output():
    z=np.arange(323).reshape(19,17)/1000;b=fixed_basis(basis(),10);base=np.zeros_like(z)
    raw=project(z,base,b)
    out,meta=prepare(raw,base,b,np.zeros(17),-np.ones(17),np.ones(17),lambda x:True,'legacy_prepare')
    expected=project(np.clip(smooth(raw,np.zeros(17)), -1,1),base,b)
    assert np.array_equal(out,expected) and not np.array_equal(out,raw)

def test_unknown_inflight_never_replayed(tmp_path):
    atomic_json(tmp_path/'ledger.json',{'inflight':{'id':'unknown'}})
    with pytest.raises(ValueError,match='UNKNOWN_WORK'):assert_resume_safe(tmp_path)

def test_full_execution_equations_are_retained_with_logging():
    # Filter side-effect-free evidence statements, then compare the physical stepping loop core.
    old=ast.parse((ROOT/'ff_prior/physical_audit.py').read_text(encoding='utf-8'))
    new=ast.parse((ROOT/'ff_prior_rep_fidelity/physical_audit.py').read_text(encoding='utf-8'))
    def calls(tree,name):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        return [ast.dump(n,include_attributes=False) for n in ast.walk(fn) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr in ('mj_step','forward','reset')]
    assert calls(old,'replay')==calls(new,'replay')
    assert calls(old,'validate')==calls(new,'validate')

def test_bounded_no_learning_or_data_mutation():
    assert len(P['ranks'])*len(P['modes'])==12
    assert P['repair_iterations']==0 and not P['training'] and not P['TEST'] and P['TRAIN_added']==0
    assert P['coupled_cap']>=12*len(P['validation_intervals'])
