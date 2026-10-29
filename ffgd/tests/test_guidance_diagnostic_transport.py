"""Software fixture only: diagnostic reproduction must honor the source policy."""
import json
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from ffgd.diffusion import sample,SamplingBudget
from ffgd.model import ConditionalDenoiser
from ffgd.calibration_guidance import transition_check,_recorded_transport


@pytest.mark.parametrize('transport',['reconstruct_noise','retain_network_noise'])
def test_transition_diagnostic_reproduces_each_source_policy(tmp_path,monkeypatch,transport):
    torch.set_num_threads(1);torch.manual_seed(74)
    model=ConditionalDenoiser(max_free=20,hidden=16,layers=1,heads=2).eval()
    condition=dict(tokens=torch.zeros(1,3,32))
    result=sample(model,condition,13,11,SamplingBudget(steps=6,guide_steps=3,guidance_transport=transport),
                  lambda x:dict(total=(x-.3).square().mean()))
    source=tmp_path/'candidate';source.mkdir();output=tmp_path/'diagnostic';output.mkdir()
    trace=result['trace'];keys=('input','output','clean','guided')
    np.savez_compressed(source/'denoising.npz',**{key:np.stack([r[key] for r in trace]) for key in keys})
    metadata=[{k:v for k,v in r.items() if k not in keys} for r in trace]
    (source/'denoising.json').write_text(json.dumps(metadata),encoding='utf-8')
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    monkeypatch.setattr('ffgd.conditioning.encode_condition',lambda *args:np.zeros((3,32)))
    monkeypatch.setattr('ffgd.model_io.normalize_condition',lambda *args:condition)
    checkpoint=dict(mean=np.zeros(13),scale=np.ones(13))
    manifest=dict(budget=dict(sampling=dict(guidance_transport=transport)))
    energy=SimpleNamespace(native_evaluate=lambda z:dict(total=float(z@z)))
    report=transition_check(output,source,dict(transition_steps=[3,4]),manifest,model,checkpoint,None,None,None,
                            SimpleNamespace(H=torch.zeros(6)),energy)
    assert report['passed'] and report['next_network_calls']==4 and report['main_candidate_count_added']==0
    assert all(r['guidance_transport']==transport for r in report['rows'])
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    # The alternate transition is a counterfactual, not a source reproduction.
    for row in report['rows']:
        other='retain_network_noise' if transport=='reconstruct_noise' else 'reconstruct_from_guided'
        assert row['predictions'][other]['saved_output_max_difference']>1e-5


def test_old_trace_default_and_mismatched_policy_rejection():
    assert _recorded_transport(dict(budget=dict(sampling={})),{})=='reconstruct_noise'
    with pytest.raises(ValueError,match='MISMATCH'):
        _recorded_transport(dict(budget=dict(sampling=dict(guidance_transport='retain_network_noise'))),
                            dict(guidance_transport='reconstruct_noise'))
    with pytest.raises(ValueError,match='UNKNOWN'):
        _recorded_transport(dict(budget=dict(sampling=dict(guidance_transport='unregistered'))),{})


def test_diagnostic_secant_action_preserves_spd_and_secant_equation():
    from ffgd.calibration_guidance import _one_secant_direction
    torch.manual_seed(75);a=torch.randn(9,9,dtype=torch.float64)
    h=a.T@a+torch.eye(9,dtype=torch.float64);s=torch.randn(9,dtype=torch.float64);y=h@s
    base=torch.diag(torch.linspace(1,20,9,dtype=torch.float64));factor=torch.linalg.cholesky(base)
    # Secant equation and positive descent verify mathematical behavior.
    result,applied,_=_one_secant_direction(factor,y,s,y)
    assert applied;torch.testing.assert_close(result,s,rtol=1e-12,atol=1e-12)
    for _ in range(3):
        g=torch.randn(9,dtype=torch.float64);d,applied,_=_one_secant_direction(factor,g,s,y)
        assert applied and g@d>0
    rejected,applied,_=_one_secant_direction(factor,y,s,-y)
    assert not applied;torch.testing.assert_close(rejected,torch.linalg.solve(base,y))
