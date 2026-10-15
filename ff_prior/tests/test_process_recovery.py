import os
import subprocess
import sys
import pytest
from ff_prior.storage import Ledger,RunLock
from ff_prior.recovery import recover_dead_lock


def test_real_crash_preserves_budget_and_duplicate_results(tmp_path):
    code="from ff_prior.storage import RunLock,Ledger; import os,sys; lock=RunLock(sys.argv[1]); lock.__enter__(); ledger=Ledger(sys.argv[1],{'test':'crash'}); ledger.begin('first','main',100,2); ledger.finish({'status':'MODEL_VALID'},{'distance':19}); ledger.begin('second','main',100,2); os._exit(7)"
    child=subprocess.run([sys.executable,"-c",code,str(tmp_path)],capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    assert child.returncode==7 and (tmp_path/"run.lock").exists()
    assert recover_dead_lock(tmp_path)["status"]=="DEAD_OWNER_LOCK_RECOVERED"
    with RunLock(tmp_path):
        ledger=Ledger(tmp_path,{"test":"crash"})
        assert ledger.data["calls"]["main.distance"]==19 and ledger.data["attempts"]["main"]==2
        assert ledger.data["completed"]["second"]["status"]=="INTERRUPTED"
        assert not ledger.begin("first","main",100,2)
        with pytest.raises(RuntimeError,match="BUDGET_EXHAUSTED"): ledger.begin("third","main",100,2)


def test_real_concurrent_worker_cannot_acquire_or_remove_live_lock(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(RuntimeError,match="alive"): recover_dead_lock(tmp_path)
        code="from ff_prior.storage import RunLock; import sys; lock=RunLock(sys.argv[1]); lock.__enter__()"
        child=subprocess.run([sys.executable,"-c",code,str(tmp_path)],capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        assert child.returncode!=0 and b"FileExistsError" in child.stderr
