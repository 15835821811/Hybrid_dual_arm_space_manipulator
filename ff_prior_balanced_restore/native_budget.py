"""Durable phase reservations and strict guards around each new native step.

The saved reservation is an upper bound, never an executed-step claim. Observed
counts are checkpointed every 250 completed steps and at every phase exit. A
crashed phase cannot be resumed: its last reservation remains conservative.
"""
from pathlib import Path
import time
from ff_prior_pilot_train.storage import atomic_json


class NativeBudgetError(RuntimeError):
    pass


class NativeBudget:
    def __init__(self, out, deadline, steps_each=13750, total_cap=27500, on_check=None):
        self.path = Path(out)/'native_budget.json'
        if self.path.exists():
            raise NativeBudgetError('UNKNOWN_NATIVE_WORK_NOT_REPLAYABLE')
        if int(steps_each) != steps_each or int(total_cap) != total_cap or steps_each < 1 or total_cap < 1:
            raise ValueError('positive integral native limits required')
        self.deadline = float(deadline)
        self.steps_each, self.total_cap = int(steps_each), int(total_cap)
        self.on_check = on_check
        self.reserved = dict(execution=0, replay=0)
        self.attempted = dict(execution=0, replay=0)
        self.completed = dict(execution=0, replay=0)
        self.phase = None
        self.started = set()
        self.finished = set()
        self.pending = None
        self.save()

    def check(self):
        if time.time() >= self.deadline:
            raise NativeBudgetError('NATIVE_WALL_BUDGET')
        if self.on_check is not None:
            try:
                self.on_check()
            except RuntimeError as error:
                raise NativeBudgetError('NATIVE_PARENT_BUDGET: '+str(error)) from error

    def begin_phase(self, kind, steps):
        self.check()
        if kind not in self.reserved or kind in self.started or self.phase is not None or self.pending is not None:
            raise NativeBudgetError('NATIVE_PHASE_REENTRY_OR_UNRESOLVED_STEP')
        if int(steps) != steps or steps < 0 or steps > self.steps_each:
            raise NativeBudgetError('NATIVE_PHASE_STEP_BUDGET')
        if sum(self.reserved.values())+steps > self.total_cap:
            raise NativeBudgetError('NATIVE_TOTAL_RESERVATION_BUDGET')
        if kind == 'replay' and ('execution' not in self.finished or steps > self.completed['execution']):
            raise NativeBudgetError('REPLAY_MUST_MATCH_RECORDED_EXECUTION_PREFIX')
        self.reserved[kind] = int(steps)
        self.started.add(kind)
        self.phase = kind
        # This entire phase is durably reserved before any native step.
        self.save()

    def before_step(self, kind):
        self.check()
        if self.phase != kind or self.pending is not None:
            raise NativeBudgetError('NATIVE_PHASE_OR_PENDING_STEP_MISMATCH')
        if self.attempted[kind] >= self.reserved[kind] or self.attempted[kind] >= self.steps_each:
            raise NativeBudgetError('NATIVE_PHASE_STEP_BUDGET')
        if sum(self.attempted.values()) >= self.total_cap:
            raise NativeBudgetError('NATIVE_TOTAL_STEP_BUDGET')
        self.attempted[kind] += 1
        self.pending = dict(phase=kind, index=self.attempted[kind]-1)

    def after_step(self, kind):
        if self.phase != kind or self.pending != dict(phase=kind, index=self.completed[kind]):
            raise NativeBudgetError('NATIVE_STEP_COMPLETION_MISMATCH')
        self.completed[kind] += 1
        self.pending = None
        if self.completed[kind] % 250 == 0:
            self.save()

    def finish_phase(self, kind):
        if self.phase != kind:
            raise NativeBudgetError('NATIVE_PHASE_FINISH_MISMATCH')
        if self.pending is None:
            self.finished.add(kind)
            self.phase = None
        # Preserve an unresolved attempt. Do not call it a completed step.
        self.save()

    def snapshot(self):
        return dict(schema='native_phase_reservation_v1', reserved_upper_bound=self.reserved.copy(),
            attempted_observed=self.attempted.copy(), completed_observed=self.completed.copy(),
            limits=dict(steps_each=self.steps_each, total=self.total_cap),
            phase=self.phase, pending=self.pending, started=sorted(self.started), finished=sorted(self.finished),
            crash_policy='no resume or replay of unknown work; reserved values bound uncheckpointed attempts',
            checkpoint_every_completed_steps=250, counters_are_not_model_intervals=True)

    def save(self):
        value = self.snapshot()
        atomic_json(self.path, value)
        return value
