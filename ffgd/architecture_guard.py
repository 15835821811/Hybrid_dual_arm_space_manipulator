"""Runtime tripwire for forbidden legacy planners and path restoration calls."""
import sys
import importlib.abc


PREFIXES=('ff_prior','v6_lite','tracking_diffusion','ffgd.baselines')


class LegacySentinel(importlib.abc.MetaPathFinder):
    def __init__(self):self.calls=0;self.modules=set();self.blocked=[]

    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(PREFIXES):
            self.blocked.append(fullname);raise RuntimeError('LEGACY_IMPORT_SENTINEL: '+fullname)
        return None

    def profile(self,frame,event,arg):
        if event!='call':return
        name=frame.f_globals.get('__name__','')
        if name.startswith(PREFIXES):
            self.blocked.append(name+'.'+frame.f_code.co_name);raise RuntimeError('LEGACY_CALL_SENTINEL: '+self.blocked[-1])
        if name.startswith('ffgd.'):
            self.calls+=1;self.modules.add(name)

    def __enter__(self):
        self.previous=sys.getprofile();sys.meta_path.insert(0,self);sys.setprofile(self.profile);return self

    def __exit__(self,*args):
        sys.setprofile(self.previous);sys.meta_path.remove(self)

    def report(self):return dict(forbidden_calls=self.blocked,observed_ffgd_calls=self.calls,modules=sorted(self.modules))
