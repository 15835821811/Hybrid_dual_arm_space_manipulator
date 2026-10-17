"""Bound runtime retention without changing models, initial states, or gates."""
from collections import OrderedDict
import ctypes
from ctypes import wintypes
import gc
import os
from ff_prior.storage import digest
from ff_prior.scenes import SceneSpec
from ff_prior.contracts import assert_task_compatible
from ff_prior_origin_scene.runtime import at_scene_origin


def memory_usage():
    if os.name!='nt':raise RuntimeError('memory guard requires audited Windows host')
    class ProcessMemory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
            (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage',
                'PeakPagefileUsage','PrivateUsage')]
    class SystemMemory(ctypes.Structure):
        _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD)]+[
            (name,ctypes.c_ulonglong) for name in ('ullTotalPhys','ullAvailPhys','ullTotalPageFile',
                'ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]
    p=ProcessMemory();p.cb=ctypes.sizeof(p)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(-1),ctypes.byref(p),p.cb):raise ctypes.WinError()
    s=SystemMemory();s.dwLength=ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):raise ctypes.WinError()
    return {'private_bytes':int(p.PrivateUsage),'working_bytes':int(p.WorkingSetSize),
            'available_physical_bytes':int(s.ullAvailPhys),'total_physical_bytes':int(s.ullTotalPhys)}


class TaskOriginSceneCache:
    def __init__(self,canonical,initial,profile,max_entries=64,max_scene_models=4,
                 max_private_bytes=16*1024**3,min_available_bytes=2*1024**3):
        if max_entries<1 or max_scene_models<1:raise ValueError('positive cache quotas required')
        self.canonical,self.initial,self.profile=canonical,initial,profile
        self.max_entries,self.max_scene_models=max_entries,max_scene_models
        self.max_private_bytes,self.min_available_bytes=max_private_bytes,min_available_bytes
        self.cache=OrderedDict();self.hits=self.builds=self.evictions=0
        self.peak_entries=self.peak_scene_models=self.peak_private_bytes=0

    def scene_keys(self):
        return [k for k,v in self.cache.items() if v[0].model is not self.canonical.model]

    def evict(self,key):
        scene=self.cache[key][0].model is not self.canonical.model
        del self.cache[key];self.evictions+=1
        if scene:gc.collect()

    def guard(self):
        m=memory_usage();self.peak_private_bytes=max(self.peak_private_bytes,m['private_bytes'])
        if m['private_bytes']>self.max_private_bytes or m['available_physical_bytes']<self.min_available_bytes:
            raise RuntimeError('RUNTIME_MEMORY_BUDGET_EXHAUSTED: '+str(m))
        return m

    def get(self,row):
        task=row['task'];scene=SceneSpec.from_dict(row.get('scene',{}))
        key=digest({'initial':task['contract']['initial_state_id'],'robot':task['contract']['robot_hash'],
                    'scene':scene.to_dict(),'q0':task['condition']['initial_features']})
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key)
        else:
            # Release retained compiled meshes before allocating another scene.
            while len(self.cache)>=self.max_entries:self.evict(next(iter(self.cache)))
            while scene.objects and len(self.scene_keys())>=self.max_scene_models:self.evict(self.scene_keys()[0])
            self.guard()
            runtime=at_scene_origin(self.canonical,self.initial,scene,task['condition']['initial_features'])
            rr,ii,cc,_=runtime;assert_task_compatible(task,rr,ii,cc,self.profile)
            self.cache[key]=runtime;self.builds+=1
            self.peak_entries=max(self.peak_entries,len(self.cache))
            self.peak_scene_models=max(self.peak_scene_models,len(self.scene_keys()))
            self.guard()
        rr,ii,cc,proof=self.cache[key]
        assert_task_compatible(task,rr,ii,cc,self.profile)
        return rr,ii,cc,proof

    def stats(self):
        return {'cache_entries':len(self.cache),'scene_models':len(self.scene_keys()),'builds':self.builds,
            'hits':self.hits,'evictions':self.evictions,'peak_entries':self.peak_entries,
            'peak_scene_models':self.peak_scene_models,'peak_private_bytes':self.peak_private_bytes,
            'limits':{'entries':self.max_entries,'scene_models':self.max_scene_models,
                      'private_bytes':self.max_private_bytes,'minimum_available_bytes':self.min_available_bytes},
            'memory':memory_usage(),'physics_contract_changed':False,
            'scope':'retained runtime cache; callers may temporarily hold an evicted runtime'}
