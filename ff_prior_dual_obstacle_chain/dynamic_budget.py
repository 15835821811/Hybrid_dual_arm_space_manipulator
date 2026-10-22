import time
from ff_prior_pilot_train.storage import atomic_json
from ff_prior_origin_learn_bounded.runtime import memory_usage
from .protocol import P

class Budget:
    def __init__(self,out):
        self.out=out;self.deadline=time.time()+P['wall_s'];self.counts={'native':0,'replay':0};self.pending=None
    def check(self):
        if time.time()>=self.deadline:raise RuntimeError('WALL_BUDGET')
        if memory_usage()['private_bytes']>P['max_bytes']:raise RuntimeError('MEMORY_BUDGET')
        if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>P['max_bytes']-128*1024**2:raise RuntimeError('OUTPUT_BUDGET')
    def step(self,kind):
        if sum(self.counts.values())>=P['max_native_replay_steps']:raise RuntimeError('NATIVE_BUDGET')
        if self.counts[kind]>=P['steps_each']:raise RuntimeError('PHASE_STEP_BUDGET')
        self.counts[kind]+=1
    def save(self):
        atomic_json(self.out/'budget.json',dict(counts=self.counts,pending=self.pending,limits=P,extra_coupled=0))
