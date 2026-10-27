"""Runtime tripwire for forbidden legacy planners and path restoration calls."""
import sys
import importlib.abc
import os
from pathlib import Path
import hashlib
import marshal


PREFIXES=('ff_prior','v6_lite','tracking_diffusion','ffgd.baselines')
_audit_owner=None
_audit_installed=False
_data_extensions={'.npz','.npy','.pt','.pth','.pkl','.pickle','.json','.yaml','.yml','.csv','.h5','.hdf5','.zip'}


def _file_audit(event,args):
    owner=_audit_owner
    if owner is None or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    filename,mode,flags=args
    reading=('r' in mode or '+' in mode) if isinstance(mode,str) else (flags & os.O_ACCMODE)!=os.O_WRONLY
    if not reading:return
    resolved=os.path.normcase(os.path.abspath(os.fsdecode(filename)))
    owner.reads.add(resolved)
    if owner.allowed_data_reads is not None and Path(resolved).suffix.lower() in _data_extensions and resolved not in owner.allowed_data_reads:
        owner.blocked.append('data_read:'+resolved)
        raise RuntimeError('UNDECLARED_PLANNER_DATA_READ: '+resolved)


class LegacySentinel(importlib.abc.MetaPathFinder):
    def __init__(self,allowed_data_reads=None):
        self.calls=0;self.modules=set();self.blocked=[];self.reads=set();self.methods={}
        self.allowed_data_reads=None if allowed_data_reads is None else {os.path.normcase(os.path.abspath(p)) for p in allowed_data_reads}

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
            key=(name,frame.f_code.co_name,frame.f_code.co_filename,frame.f_code.co_firstlineno)
            if key not in self.methods:self.methods[key]=hashlib.sha256(marshal.dumps(frame.f_code)).hexdigest()

    def __enter__(self):
        global _audit_owner,_audit_installed
        if _audit_owner is not None:raise RuntimeError('nested planner sentinel is unsupported')
        if not _audit_installed:sys.addaudithook(_file_audit);_audit_installed=True
        _audit_owner=self
        self.previous=sys.getprofile();sys.meta_path.insert(0,self);sys.setprofile(self.profile);return self

    def __exit__(self,*args):
        global _audit_owner
        _audit_owner=None
        sys.setprofile(self.previous);sys.meta_path.remove(self)

    def report(self):
        identities=[]
        for name in sorted(self.modules):
            source=Path(sys.modules[name].__file__)
            identities.append(dict(module=name,path=str(source.resolve()),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        return dict(forbidden_calls=self.blocked,observed_ffgd_calls=self.calls,modules=sorted(self.modules),
                    module_identities=identities,actual_methods=[dict(module=m,function=f,source=p,line=line,executed_code_sha256=self.methods[(m,f,p,line)]) for m,f,p,line in sorted(self.methods)],
                    files_read=sorted(self.reads),file_audit_enforced=self.allowed_data_reads is not None,
                    allowed_data_reads=sorted(self.allowed_data_reads) if self.allowed_data_reads is not None else None)
