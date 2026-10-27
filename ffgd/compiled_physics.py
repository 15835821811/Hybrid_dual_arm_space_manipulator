"""Optional exact-graph TorchScript experiment; disabled in the main planner.

This changes execution of the fixed robot tree graph, not its physical model.
It must pass native values and whole-path gradient regressions before enabling.
"""
import time
import torch
from .physics import tensor


KEYS=('position','rotation','com','Jcom','Jangular','omega','alpha','acom','inertia')


class TreeKernel(torch.nn.Module):
    def __init__(self,tree):
        super().__init__();self.function=tree.kinematics

    def forward(self,qpos,velocity,acceleration):
        values=self.function(qpos,velocity,acceleration)
        return tuple(values[k] for k in KEYS)


def compile_tree(tree,output=None):
    start=time.perf_counter();r=tree.robot;d=r.new_data();x=tensor(d.qpos);v=torch.zeros(r.model.nv,dtype=torch.float64)
    compiled=torch.jit.trace(TreeKernel(tree).eval(),(x,v,v),check_trace=False)
    compiled=torch.jit.freeze(compiled.eval())
    if output is not None:torch.jit.save(compiled,str(output))
    original=tree.kinematics
    def kinematics(qpos,velocity=None,acceleration=None):
        vel=torch.zeros(tree.nv,dtype=qpos.dtype) if velocity is None else velocity
        acc=torch.zeros_like(vel) if acceleration is None else acceleration
        return dict(zip(KEYS,compiled(qpos,vel,acc)))
    tree.kinematics=kinematics
    return dict(seconds=time.perf_counter()-start,model_hash=r.model_hash,original=original,compiled=compiled)
