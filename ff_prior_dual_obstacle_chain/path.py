"""C2 detours between fixed world waypoints; no endpoint or timing relabeling."""
import numpy as np
class BypassPath:
    def __init__(self,original,offset):
        self.original=original;self.offset=np.asarray(offset,dtype=float)
        self.target_rotation_world=original.target_rotation_world
        self.start=float(original.transition_duration_s)
        self.durations=np.asarray(original.segment_durations_s)
        self.boundaries=np.r_[self.start,self.start+np.cumsum(self.durations)]
    def sample(self,t):
        p,v=self.original.sample(t)
        if t<self.start or t>=self.boundaries[-1]:return p,v
        i=min(len(self.durations)-1,int(np.searchsorted(self.boundaries,t,side='right')-1))
        s=(t-self.boundaries[i])/self.durations[i]
        b=64*s**3*(1-s)**3
        db=192*s*s*(1-s)**2*(1-2*s)/self.durations[i]
        return p+b*self.offset,v+db*self.offset
