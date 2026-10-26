"""Continue full-data training after the registered producer, without manual gates."""
import os
for _k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_k]="1"
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time
from .pipeline import write_json
from .round2_data import BASE,build,read


def process_alive(pid):
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle: return False
    code=ctypes.c_ulong()
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    okay=kernel.GetExitCodeProcess(handle,ctypes.byref(code))
    kernel.CloseHandle(handle)
    return bool(okay and code.value==259)


def main():
    producer_pid=int((BASE/"produce.pid").read_text().strip())
    while True:
        summary=read(BASE/"teachers/summary.json")
        if summary.get("production_complete") and (BASE/"mid_exit.json").exists(): break
        if not process_alive(producer_pid):
            raise RuntimeError("Registered producer stopped before completion; resume production first")
        time.sleep(5)
    if read(BASE/"mid_exit.json")["returncode"]:
        raise RuntimeError("Midpoint training failed; inspect mid.stdout.log")
    dataset=build(BASE,500)
    for name,fraction in (("train_t1",0.),("train_t2",.2)):
        output=BASE/name
        if (output/"model.pt").exists(): continue
        write_json(BASE/"training_stage.json",{"stage":name,"dataset":str(dataset),"steps":30000,"startup_fraction":fraction,"status":"running"})
        with (BASE/(name+".stdout.log")).open("a",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-m","tracking_diffusion.paper_model","--dataset",str(dataset),
                "--output",str(output),"--steps","30000","--startup-fraction",str(fraction),"--device","cuda"],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(BASE/"training_stage.json",{"stage":"T1_T2_complete","status":"complete","dataset":str(dataset)})


if __name__=="__main__": main()
