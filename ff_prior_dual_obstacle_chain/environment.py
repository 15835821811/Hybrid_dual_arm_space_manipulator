from ff_prior_terminal_tracking.environment import Environment as Original
from .path import BypassPath
class Environment(Original):
    def __init__(self,robot,terminal,contract,budget):
        # The original initializer records t=0, at which the new curve is identical.
        super().__init__(robot,terminal,contract,budget)
        self.path=BypassPath(self.path,contract['bypass_offset_m'])
