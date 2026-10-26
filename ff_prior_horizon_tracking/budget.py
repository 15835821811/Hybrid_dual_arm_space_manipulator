import time
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_origin_learn_bounded.runtime import memory_usage
from .protocol import P

class Budget:
    def __init__(self,out):
        self.out=out;self.started=time.time();self.deadline=self.started+P['wall_s']
        self.physics_deadline=self.deadline-P['media_reserve_s']
        self.counts={'native':0,'replay':0};self.pending=None;self.last_disk_check=0.
    def check(self,phase='physics'):
        if time.time()>=(self.physics_deadline if phase=='physics' else self.deadline):
            raise RuntimeError('WALL_BUDGET')
        if memory_usage()['private_bytes']>P['max_bytes']:raise RuntimeError('MEMORY_BUDGET')
        if time.time()-self.last_disk_check>5:
            self.last_disk_check=time.time()
            if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>P['max_bytes']-256*1024**2:
                raise RuntimeError('OUTPUT_BUDGET')
    def step(self,kind):
        if time.time()>=self.physics_deadline:raise RuntimeError('PHYSICS_WALL_BUDGET')
        if sum(self.counts.values())>=P['max_native_replay_steps']:raise RuntimeError('NATIVE_BUDGET')
        if self.counts[kind]>=P['steps_each']:raise RuntimeError('PHASE_STEP_BUDGET')
        self.counts[kind]+=1
    def save(self):
        atomic_json(self.out/'budget.json',dict(counts=self.counts,pending=self.pending,limits=P,
            elapsed_s=time.time()-self.started,extra_coupled=0))
