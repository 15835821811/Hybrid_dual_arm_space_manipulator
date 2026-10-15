"""Package compact measured repair evidence; never copy weights or native arrays."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import zipfile
import yaml
from ff_prior.storage import read_json, atomic_json, file_digest
from ff_prior.pipeline import source_manifest


def main():
    base=Path('artifacts/v7_1')
    out=Path('ff_prior/evidence/diffusion_repair')
    out.mkdir(parents=True,exist_ok=True)
    old=Path('ff_prior/evidence/smoke_002')
    original=read_json(old/'evidence_manifest.json')
    assert all(file_digest(old/name)==h for name,h in original['files'].items())
    refs={r:subprocess.check_output(['git','rev-parse',r],text=True).strip()
          for r in ('main','v6.4-a-diffusion-planning')}
    assert refs=={'main':'84faf3063faeeaf5914e5acf4bacca55d0f02e80',
        'v6.4-a-diffusion-planning':'070e7008923bf2642dcecf9806745564388669b9'}
    expert=subprocess.check_output(['git','-C','E:/v70expert','status','--short'],text=True,encoding='utf-8').strip()
    assert expert==(base/'baseline/expert_status.txt').read_text(encoding='utf-8-sig').strip()
    original_config=yaml.safe_load(subprocess.check_output(
        ['git','show','f2284b64:configs/ff_prior/v7_1.proposed.yaml'],text=True,encoding='utf-8'))
    current_config=yaml.safe_load(Path('configs/ff_prior/v7_1.proposed.yaml').read_text(encoding='utf-8'))
    for config in (original_config,current_config):
        config.pop('training'); config.pop('sampling')
    assert current_config==original_config
    for name in ('diagnosis_before.json','diagnosis_numerics_only.json','diagnosis_final.json',
                 'dataset_audit.json','comparison_audit.json','diagnosis.png','resume_before.json','tests_final.xml'):
        shutil.copy2(base/'diffusion_repair'/name,out/name)
    for name in ('smoke_diffusion_repair_001','smoke_diffusion_repair_002'):
        run=base/name
        manifest=read_json(run/'run.json')
        with zipfile.ZipFile(run/'executed_source.zip') as archive:
            assert all(hashlib.sha256(archive.read(p)).hexdigest()==h
                       for p,h in manifest['source_hashes'].items())
        if name.endswith('002'):
            assert source_manifest()==manifest['source_hashes']
        files=[run/p for p in ('summary.json','benchmark.json','run.json','ledger.json',
               'benchmark_tasks.json','dataset.json','scene.json','pair_policy.json',
               'learning/training.json','learning/scaler.json','executed_source.zip')]
        if (run/'learning/clean_prior.json').exists(): files.append(run/'learning/clean_prior.json')
        files+=list((run/'benchmark').glob('*/planning.json'))
        files+=list((run/'benchmark').glob('*/execution/report.json'))
        files+=list((run/'benchmark').glob('*/execution/frozen_task.json'))
        for file in files:
            dest=out/name/file.relative_to(run)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,dest)
    ref=Path('C:/Users/admin/AppData/Local/Temp/v71_htdcr_reference_20261015')
    commit=subprocess.check_output(['git','-C',str(ref),'rev-parse','HEAD'],text=True).strip()
    assert commit=='900d781ba63f0123dff95e49a76d51cb49a74803'
    paths=['scripts/training/generate_training_data.py','scripts/training/refine_training_data.py',
           'scripts/training/train_proposed.py','configs/method/proposed.yaml']
    paths+=['src/htdcr_planning/'+p for p in ('planning/diffusion_planner.py',
        'interface/guided_sampler.py','models/DiffusionProcess.py','robot_model/kinematics.py',
        'dataset/dataloaderConfig.py')]
    atomic_json(out/'reference_provenance.json',{'repository':'https://github.com/zhanglooong/htdcr-actuator-aware-diffusion-planning',
        'commit':commit,'scope':'read-only source comparison; no training or generation executed',
        'files':{p:file_digest(ref/p) for p in paths}})
    atomic_json(out/'preservation.json',{'old_refs':refs,'original_evidence_hashes_preserved':True,
        'old_expert_worktree_status_preserved':True,'physical_config_outside_learning_unchanged':True,
        'user_patch_preserved':read_json(out/'comparison_audit.json')['user_tracked_changes_preserved'],
        'old_evidence_manifest_sha256':file_digest(old/'evidence_manifest.json')})
    files={p.relative_to(out).as_posix():file_digest(p) for p in sorted(out.rglob('*'))
           if p.is_file() and p.name!='manifest.json'}
    atomic_json(out/'manifest.json',{'scope':'SMOKE_DEV_RECHECK','checkpoint_included':False,
        'large_data_included':False,'pilot':'NOT_RUN','TEST':'NOT_RUN','files':files})
    print({'files':len(files),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
           'old_refs_and_evidence_preserved':True})


if __name__=='__main__': main()
