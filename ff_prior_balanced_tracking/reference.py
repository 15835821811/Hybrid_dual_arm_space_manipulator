"""Frozen geometric C2 detour around three public segment-midpoint obstacles.

This changes only the position reference between original waypoints. It never
changes a waypoint, its rotation or arrival time and never writes actual state.
"""
import numpy as np
from .protocol import P


def bypass(position, velocity, local_time, waypoint_contract):
    position=np.asarray(position,dtype=float).copy()
    velocity=np.asarray(velocity,dtype=float).copy()
    start=float(waypoint_contract['transition_duration_s'])
    elapsed=float(local_time)-start
    durations=np.asarray(waypoint_contract['segment_durations_s'],dtype=float)
    boundaries=np.r_[0.,np.cumsum(durations)]
    if elapsed<0. or elapsed>=float(waypoint_contract['path_duration_s']):
        return position,velocity
    segment=int(np.searchsorted(boundaries,elapsed,side='right')-1)
    if segment not in P['bypass_segments']:
        return position,velocity
    u=float(np.clip((elapsed-boundaries[segment])/durations[segment],0.,1.))
    profile=64.*u**3*(1.-u)**3
    rate=192.*u**2*(1.-u)**2*(1.-2.*u)/durations[segment]
    delta=P['bypass_amplitude_m']*np.asarray(P['bypass_axis_world'])
    return position+profile*delta,velocity+rate*delta
