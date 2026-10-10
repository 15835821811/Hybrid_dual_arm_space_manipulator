"""Explicit C4-A P2 schedule; C3 defaults and raw qualifier remain unchanged."""
import copy

from .route_initializers import FrozenSeedInitializer, json_raw
from .route_optimizer_protocol import digest, initial_candidates, SearchSpec


class RulePreservingPortfolio:
    """Four unchanged rule seeds, frozen A/v1 and B/v2, then two polls.

    The optimizer counts duplicates/rejections as slots only for this schedule.
    initializer_slot identifies the original raw validation condition, not the
    new physical slot index (4/5). Construction never samples a model.
    """
    def __init__(self, proposals):
        frozen=FrozenSeedInitializer(proposals)
        self._proposals=copy.deepcopy(frozen._overrides)
        if any(p.get('source')!='diffusion' for p in self._proposals.values()):
            raise ValueError('C4-A P2 accepts only frozen diffusion proposals')
        self.identity=dict(schema='v64_c4a_rule_preserving_portfolio_v1',
            overrides_sha256=digest(json_raw(proposals)),rule_positions=[0,1,2,3],
            learned_positions={'4':'A/v1','5':'B/v2'},poll_positions=[6,7],
            duplicate_consumes_slot=True,raw_rejection_consumes_slot=True,
            candidate_budget=8,extra_physical_calls=0)

    def __call__(self,task):
        return [*initial_candidates(task),
                {**copy.deepcopy(self._proposals[1]),'initializer_slot':1},
                {**copy.deepcopy(self._proposals[3]),'initializer_slot':3}]


def optimize_strategy(strategy, task, preferences, identity, evaluator, root, *, proposals=None):
    from .continuous_route_optimizer import optimize
    if strategy not in ('P0','P1','P2'):
        raise ValueError('only predeclared C4-A strategies')
    if (strategy=='P0') != (proposals is None):
        raise ValueError('P0 has no proposals; P1/P2 require exactly two frozen proposals')
    return optimize(task,preferences,identity,evaluator,root,search_spec=SearchSpec(candidate_budget=8),
        initializer=FrozenSeedInitializer(proposals) if strategy=='P1' else None,
        seed_schedule=RulePreservingPortfolio(proposals) if strategy=='P2' else None)
