"""Reservations precede work. Counts include failed/aborted evaluations."""
import time
from pathlib import Path
from datetime import datetime
from ff_prior.storage import read_json
from ff_prior_pilot_train.storage import atomic_json
from .protocol import P, PACKAGE


class Limit(RuntimeError):
    pass


class Budget:
    def __init__(self, out, started=None):
        self.out=Path(out)
        auth=read_json(PACKAGE/'evidence/authorization.json') if started is None else {}
        stamp=auth.get('started_at',auth.get('authorized_at'))
        self.started=float(started) if started is not None else (float(stamp) if isinstance(stamp,(float,int)) else datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp())
        self.deadline=self.started+P['wall_s']
        self.physics_deadline=self.deadline
        self.used=dict(coupled=0,intervals=0,native=0,neural=0,geometry=0,mass_kinematic=0)
        self.completed=dict(self.used);self.events=[];self.stage=None
        self.stage_wall={};self.stage_start=time.monotonic()

    def check(self, output=0):
        if time.time()>=self.deadline:raise Limit('UNIT_WALL_CAP')
        if self.stage is not None:
            caps=dict(planner=P['planner_s'],native=P['native_s'],media=5400,review=5400)
            if self.stage_wall.get(self.stage,0.)+time.monotonic()-self.stage_start>=caps[self.stage]:raise Limit('CATEGORY_WALL_CAP_'+self.stage)
        size=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) if output else 0
        if output and size+output>P['bytes_cap']:raise Limit('UNIT_STORAGE_CAP')

    def ensure_output(self,n):self.check(int(n))

    def enter(self,stage):
        if self.stage is not None:self.stage_wall[self.stage]=self.stage_wall.get(self.stage,0.)+time.monotonic()-self.stage_start
        self.stage=stage;self.stage_start=time.monotonic();self.check()

    def reserve(self,kind,n=1):
        self.check()
        cap=P[dict(coupled='coupled_cap',intervals='interval_cap',native='native_cap',neural='neural_cap',geometry='geometry_cap',mass_kinematic='mass_kinematic_cap')[kind]]
        if n<0 or self.used[kind]+n>cap:raise Limit('UNIT_'+kind.upper()+'_CAP')
        self.used[kind]+=int(n)

    def complete(self,kind,n=1):
        if self.completed[kind]+n>self.used[kind]:raise AssertionError('UNRESERVED_WORK')
        self.completed[kind]+=int(n)

    def propagation(self,n):
        # Check both before either mutation.
        if self.used['coupled']+1>P['coupled_cap'] or self.used['intervals']+n>P['interval_cap']:raise Limit('MODEL_BUDGET')
        self.reserve('coupled');self.reserve('intervals',n)

    def save(self):
        self.out.mkdir(parents=True,exist_ok=True)
        atomic_json(self.out/'budget.json',dict(reserved=self.used,completed=self.completed,
            started=self.started,elapsed_s=time.time()-self.started,deadline=self.deadline,
            model_intervals_are_native=False,stage=self.stage,stage_wall=self.stage_wall))


class Instrumentation:
    """Count direct library calls too; install only in the single production worker."""
    def __init__(self,budget):self.budget=budget;self.original={}
    def __enter__(self):
        import mujoco
        for name,kind in [('mj_geomDistance','geometry'),('mj_fullM','mass_kinematic'),
                          ('mj_kinematics','mass_kinematic'),('mj_forward','mass_kinematic')]:
            fn=getattr(mujoco,name);self.original[name]=fn
            def counted(*args,_fn=fn,_kind=kind,**kwargs):
                self.budget.reserve(_kind)
                result=_fn(*args,**kwargs);self.budget.complete(_kind);return result
            setattr(mujoco,name,counted)
        return self
    def __exit__(self,*args):
        import mujoco
        for name,fn in self.original.items():setattr(mujoco,name,fn)
