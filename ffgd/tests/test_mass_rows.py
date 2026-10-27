"""Exact block extraction and VJP tests for the calibrated base-row backend."""
import numpy as np
import torch
from ffgd.robot import Robot
from ffgd.physics import DifferentiableTree,tensor


def test_base_mass_rows_match_full_matrix_and_vjp_with_all_columns():
    torch.set_num_threads(1);r=Robot();tree=DifferentiableTree(r);d=r.new_data()
    rng=np.random.default_rng(291);d.qpos[r.qpos_ids]+=r.B@rng.normal(0,.07,17)
    x=tensor(d.qpos).requires_grad_();rows=r.base_dofs
    full=tree.mass(x)[rows];block=tree.mass_rows(x,rows)
    assert block.shape==(6,r.model.nv)
    torch.testing.assert_close(full,block,rtol=1e-12,atol=1e-12)
    cotangent=tensor(rng.normal(size=block.shape))
    a=torch.autograd.grad((full*cotangent).sum(),x)[0]
    b=torch.autograd.grad((block*cotangent).sum(),x)[0]
    torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
