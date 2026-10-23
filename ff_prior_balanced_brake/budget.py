"""Hard cumulative native/output/time guards; attempted and completed steps differ."""
import time
from ff_prior_pilot_train.storage import atomic_json
from .protocol import P


class Budget:
    def __init__(self,out,started=None):
        self.out=out;self.started=time.time() if started is None else float(started)
        self.deadline=self.started+P['wall_s']
        self.physics_deadline=self.deadline-P['media_reserve_s']
        self.counts={'native':0,'replay':0};self.reserved={'native':0,'replay':0}
        self.pending=None;self.last_disk_check=0.;self.inflight=None
    def check(self,phase='physics'):
        if time.time()>=(self.physics_deadline if phase=='physics' else self.deadline):
            raise RuntimeError('WALL_BUDGET')
        if time.time()-self.last_disk_check>5:
            self.last_disk_check=time.time()
            size=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
            if size>P['max_bytes']-P['output_reserve_bytes']:raise RuntimeError('OUTPUT_BUDGET')
    def reserve(self,kind):
        if self.inflight is not None:raise RuntimeError('UNRESOLVED_PREVIOUS_NATIVE_STEP')
        self.check('physics')
        if sum(self.reserved.values())>=P['max_native_replay_steps']:raise RuntimeError('NATIVE_BUDGET')
        if self.reserved[kind]>=P['steps_each']:raise RuntimeError('PHASE_STEP_BUDGET')
        self.reserved[kind]+=1;self.inflight=dict(kind=kind,index=self.reserved[kind]-1)
    def complete(self,kind):
        assert self.inflight is not None and self.inflight['kind']==kind
        self.counts[kind]+=1;self.inflight=None
    def save(self):
        atomic_json(self.out/'budget.json',dict(counts=self.counts,reserved=self.reserved,
            inflight=self.inflight,pending=self.pending,limits=P,elapsed_s=time.time()-self.started,
            extra_coupled=0,coupled_allocation_reused=False))
