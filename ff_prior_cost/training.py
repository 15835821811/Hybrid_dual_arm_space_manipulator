"""Small residual distillation; TRAIN gradients, VAL checkpoint selection only."""
import time
from pathlib import Path
import numpy as np
import torch
from ff_prior_next.learning import load as load_frozen
from ff_prior.diffusion import schedule, save_torch, weight_hash
from ff_prior.storage import atomic_json, read_json, digest, file_digest

CONTRACT = {'version': 'minimal_repaired_residual_distillation_1', 'train_targets': 30,
    'val_targets': 6, 'sanity_targets': 8, 'sanity_updates': 80, 'updates': 240,
    'seed': 20261031, 'lr': 1e-4, 'batch': 16, 'terminal_weight': 4.,
    'clean_aux_weight': 1., 'scaler': 'frozen', 'temporal_prior': 'frozen_rank6',
    'select': 'fixed three-band VAL clean MSE every 20 updates',
    'sanity_gate': 'fixed high-noise TRAIN loss <=30% initial, finite weights',
    'loss_scope': 'joint residual imitation; actual world pose diagnosed separately, not claimed as differentiable dynamics'}


def load(directory, checkpoint, robot_hash, initial_id):
    bundle = load_frozen(checkpoint, robot_hash, initial_id)
    saved = torch.load(Path(directory)/'model.pt', map_location='cpu', weights_only=False)
    if saved['contract'] != CONTRACT: raise ValueError('distillation contract mismatch')
    if saved['parent_checkpoint_sha256'] != file_digest(Path(checkpoint)/'model.pt'):
        raise ValueError('distillation parent mismatch')
    bundle[0].load_state_dict(saved['weights']); bundle[0].eval()
    return bundle


def train(directory, records, checkpoint, robot_hash, initial_id, progress, deadline):
    out = Path(directory); out.mkdir(parents=True, exist_ok=True)
    identity = digest({'records': records, 'contract': CONTRACT, 'parent': file_digest(Path(checkpoint)/'model.pt')})
    if (out/'training.json').exists():
        report = read_json(out/'training.json')
        if report['identity'] != identity: raise ValueError('training identity conflict')
        return report
    tr = [r for r in records if r['split'] == 'TRAIN']; va = [r for r in records if r['split'] == 'VAL']
    if len(tr) != 30 or len(va) != 6 or any(not r['eligible'] for r in records):
        raise RuntimeError('TRAINING_BLOCKED_REQUIRES_30_TRAIN_AND_6_VAL_VALID_TEACHERS')
    if {r['task']['task_hash'] for r in tr} & {r['task']['task_hash'] for r in va}:
        raise ValueError('TRAIN/VAL overlap')
    torch.set_num_threads(2); torch.manual_seed(CONTRACT['seed'])
    model, scaler, prior = load_frozen(checkpoint, robot_hash, initial_id)
    parent_hash = weight_hash(model); alpha = schedule()
    def tensors(rows):
        residual = np.stack([np.array(r['Z'])-np.array(r['base']) for r in rows])
        return (torch.tensor(scaler.z(residual), dtype=torch.float32),
                torch.tensor(np.stack([scaler.condition(r['task']['condition']) for r in rows]), dtype=torch.float32))
    x, c = tensors(tr); xv, cv = tensors(va)
    w = torch.ones(1,19,1); w[:,-1] = CONTRACT['terminal_weight']
    def objective(m, xx, cc, tt, noise):
        a = alpha[tt,None,None]; noisy = a.sqrt()*xx+(1-a).sqrt()*noise
        v = m(noisy,tt,cc); target = a.sqrt()*noise-(1-a).sqrt()*xx
        clean = a.sqrt()*noisy-(1-a).sqrt()*v
        return (((v-target)**2)*w).mean()+CONTRACT['clean_aux_weight']*(((clean-xx)**2)*w).mean()
    start_time = time.perf_counter(); start_cpu = time.process_time()
    # Separate throw-away sanity fit; never initialize the main fit from these weights.
    sanity_noise = torch.randn(x[:8].shape); sanity_t = torch.full((8,),999)
    optimizer = torch.optim.AdamW(model.parameters(),lr=CONTRACT['lr'])
    with torch.no_grad(): first = float(objective(model,x[:8],c[:8],sanity_t,sanity_noise))
    for step in range(CONTRACT['sanity_updates']):
        if time.time() >= deadline: raise RuntimeError('TRAINING_WALL_BUDGET_EXHAUSTED')
        model.train(); loss = objective(model,x[:8],c[:8],sanity_t,sanity_noise)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite sanity loss')
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if (step+1)%20 == 0: progress('training_sanity', updates=step+1,total=80)
    model.eval()
    with torch.no_grad(): last = float(objective(model,x[:8],c[:8],sanity_t,sanity_noise))
    sanity = {'initial_loss': first, 'final_loss': last, 'ratio': last/first,
        'status': 'PASS' if last <= .3*first else 'FAIL', 'training_ids': [r['sample_id'] for r in tr[:8]],
        'forward_calls': 82, 'claim': 'pipeline capacity sanity only; no generalization claim'}
    atomic_json(out/'sanity.json',sanity)
    if sanity['status'] != 'PASS':
        report = {'status': 'BLOCKED', 'reason': 'SMALL_TRAIN_SANITY_FAILED', 'identity': identity, 'sanity': sanity}
        atomic_json(out/'training.json',report); return report
    model, _, _ = load_frozen(checkpoint,robot_hash,initial_id)
    torch.manual_seed(CONTRACT['seed']+1)
    optimizer = torch.optim.AdamW(model.parameters(),lr=CONTRACT['lr'])
    # Fixed VAL perturbations are never used for gradients.
    noise_val = torch.randn((3,len(xv),19,17)); bands = (0,500,999)
    def validation():
        model.eval(); values = []
        with torch.no_grad():
            for n,t in zip(noise_val,bands):
                a = alpha[t]; noisy = a.sqrt()*xv+(1-a).sqrt()*n
                clean = a.sqrt()*noisy-(1-a).sqrt()*model(noisy,torch.full((len(xv),),t),cv)
                values.append(float((((clean-xv)**2)*w).mean()))
        return values
    history = []; initial_val = validation(); best = float(np.mean(initial_val)); best_step = 0
    parent_file = file_digest(Path(checkpoint)/'model.pt')
    def save(step): save_torch(out/'model.pt', {'weights': model.state_dict(), 'contract': CONTRACT,
        'parent_checkpoint_sha256': parent_file, 'identity': identity, 'step': step})
    save(0)
    for step in range(CONTRACT['updates']):
        if time.time() >= deadline: raise RuntimeError('TRAINING_WALL_BUDGET_EXHAUSTED')
        model.train(); indices = torch.randint(0,len(x),(16,)); t = torch.randint(0,1000,(16,))
        loss = objective(model,x[indices],c[indices],t,torch.randn_like(x[indices]))
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if (step+1)%20 == 0:
            values = validation(); score = float(np.mean(values))
            history.append({'update': step+1, 'train_loss': float(loss.detach()), 'val_clean_mse': values})
            if score < best: best, best_step = score, step+1; save(best_step)
            progress('residual_distillation',updates=step+1,total=240,selected_update=best_step)
    selected, _, _ = load(out,checkpoint,robot_hash,initial_id)
    clone, _, _ = load(out,checkpoint,robot_hash,initial_id)
    with torch.no_grad(): reload_error = float(abs(selected(xv,torch.zeros(len(xv),dtype=torch.long),cv)-clone(xv,torch.zeros(len(xv),dtype=torch.long),cv)).max())
    report = {'status': 'PASS' if reload_error == 0 else 'FAIL', 'identity': identity, 'contract': CONTRACT,
        'sanity': sanity, 'history': history, 'initial_val_clean_mse': initial_val, 'selected_val_mean_mse': best,
        'selected_update': best_step, 'selected_weight_hash': weight_hash(selected), 'parent_weight_hash': parent_hash,
        'weights_changed': weight_hash(selected) != parent_hash, 'checkpoint_sha256': file_digest(out/'model.pt'),
        'training_ids': [r['sample_id'] for r in tr], 'validation_ids': [r['sample_id'] for r in va],
        'training_task_hashes': [r['task']['task_hash'] for r in tr],
        'training_forward_calls': 240, 'validation_forward_calls': 39, 'reload_forward_calls': 2,
        'reload_error': reload_error, 'wall_s': time.perf_counter()-start_time, 'cpu_s': time.process_time()-start_cpu,
        'physical_training_label_level': 'MODEL_VALID_ONLY', 'DEV_TEST_gradient_reads': False,
        'benefit': 'NOT_ESTABLISHED_UNTIL_MATCHED_COMPARISON'}
    atomic_json(out/'training.json',report); return report
