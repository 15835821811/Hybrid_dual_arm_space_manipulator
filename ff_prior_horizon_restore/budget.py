"""Reservations and observed counters persist across robot instances and units."""
import time
from ff_prior_pilot_train.storage import atomic_json
from .protocol import P


from ff_prior_static_goal_seed.budget import BudgetExhausted


class Budget:
    def __init__(self, out, deadline, saved=None):
        self.out, self.deadline = out, deadline
        saved = saved or {}
        self.calls = saved.get('reserved_coupled', 0)
        self.integrations = saved.get('integration_intervals', 0)
        self.observed = saved.get('actual_coupled', 0)
        self.native_steps = saved.get('actual_native_and_replay_steps', 0)
        self.last_size_check = 0
        self.unit_start = (self.calls, self.integrations)
        self.unit_deadline = deadline

    def begin_unit(self):
        self.unit_start = (self.calls, self.integrations)
        self.unit_deadline = min(self.deadline, time.time()+P['unit_wall_s'])

    def check(self, upcoming=0):
        if time.time() >= self.deadline:
            raise BudgetExhausted('TOTAL_WALL_BUDGET')
        if time.time() >= self.unit_deadline:
            raise BudgetExhausted('UNIT_WALL_BUDGET')
        if self.calls+upcoming > P['coupled_cap']:
            raise BudgetExhausted('TOTAL_COUPLED_BUDGET')
        if self.calls-self.unit_start[0]+upcoming > P['unit_coupled_cap']:
            raise BudgetExhausted('UNIT_COUPLED_BUDGET')
        if time.time()-self.last_size_check > 20:
            if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) >= P['output_bytes_cap']-512*1024**2:
                raise BudgetExhausted('OUTPUT_BUDGET')
            self.last_size_check = time.time()

    def ensure_output(self, upcoming_bytes):
        current=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        if current+upcoming_bytes+512*1024**2>P['output_bytes_cap']:
            raise BudgetExhausted('OUTPUT_RESERVATION')

    def consume(self, intervals):
        self.check(1)
        if self.integrations+intervals > P['integration_cap']:
            raise BudgetExhausted('TOTAL_INTEGRATION_BUDGET')
        if self.integrations-self.unit_start[1]+intervals > P['unit_integration_cap']:
            raise BudgetExhausted('UNIT_INTEGRATION_BUDGET')
        self.calls += 1
        self.integrations += int(intervals)
        # Persist conservative reservation before entering any physical predictor.
        self.save()

    def observe(self, delta):
        self.observed += int(delta.get('coupled', 0))
        self.native_steps += int(delta.get('physics_steps', 0))
        if self.observed > self.calls or self.native_steps > P['native_steps_cap']:
            raise RuntimeError('OBSERVED_COUNTER_EXCEEDS_RESERVED_LIMIT')

    def save(self):
        value = {'reserved_coupled': self.calls, 'actual_coupled': self.observed,
                 'integration_intervals': self.integrations,
                 'actual_native_and_replay_steps': self.native_steps, 'limits': P}
        atomic_json(self.out/'budget.json', value)
        return value


class SliceBudget:
    """No unused quota transfers to another proposal or comparison method."""
    def __init__(self, parent, calls, intervals, deadline):
        self.parent = parent
        self.start = (parent.calls, parent.integrations)
        self.call_cap, self.interval_cap = calls, intervals
        self.deadline = min(parent.deadline, parent.unit_deadline, deadline)

    def check(self, upcoming=0):
        self.parent.check(upcoming)
        if time.time() >= self.deadline:
            raise BudgetExhausted('SLICE_WALL_BUDGET')
        if self.parent.calls-self.start[0]+upcoming > self.call_cap:
            raise BudgetExhausted('SLICE_COUPLED_BUDGET')

    def can_fit(self, queries, intervals):
        p = self.parent
        return (time.time()+30 < self.deadline
                and p.calls-self.start[0]+queries <= self.call_cap
                and p.integrations-self.start[1]+intervals <= self.interval_cap
                and p.calls+queries <= P['coupled_cap']
                and p.calls-p.unit_start[0]+queries <= P['unit_coupled_cap']
                and p.integrations+intervals <= P['integration_cap']
                and p.integrations-p.unit_start[1]+intervals <= P['unit_integration_cap'])

    def consume(self, intervals):
        self.check(1)
        if self.parent.integrations-self.start[1]+intervals > self.interval_cap:
            raise BudgetExhausted('SLICE_INTEGRATION_BUDGET')
        self.parent.consume(intervals)
