"""Prospective TRAIN gate. Never substitutes for model or execution validation."""
import numpy as np

POSITION_M = .005
ROTATION_RAD = np.pi / 180


def _arrays(cases):
    if not cases:
        raise ValueError('missing probes')
    identities = [(r['sample_id'], r['t'], r['seed']) for r in cases]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate probe identity')
    p = np.asarray([r['position_m'] for r in cases], dtype=float)
    a = np.asarray([r['rotation_rad'] for r in cases], dtype=float)
    loss = np.asarray([r['loss'] for r in cases], dtype=float)
    if p.shape != (len(cases), 2) or a.shape != p.shape:
        raise ValueError('both arm errors required')
    if not all(np.isfinite(v).all() and (v >= 0).all() for v in (p, a, loss)):
        raise ValueError('invalid probe evidence')
    recomputed = ((p/POSITION_M)**2 + (a/ROTATION_RAD)**2).sum(axis=1)/4
    if not np.allclose(loss, recomputed, rtol=1e-10, atol=1e-12):
        raise ValueError('loss does not match saved physical errors')
    worst = np.maximum(p.max(axis=1)/POSITION_M, a.max(axis=1)/ROTATION_RAD)
    return identities, loss, p.max(axis=1), a.max(axis=1), worst <= 1, worst


def compare(warm, after, local):
    wi, wl, wp, wa, wok, ww = _arrays(warm)
    ai, al, ap, aa, aok, aw = _arrays(after)
    bi, bl, *_ = _arrays(local)
    if wi != ai or wi != bi:
        raise ValueError('unpaired identities/order/noise')
    ratio = float(np.median(al / np.maximum(wl, 1e-12)))
    vs_local = float(np.median(al / np.maximum(bl, 1e-12)))
    lost = int(np.sum(wok & ~aok))
    gates = {
        'legacy_loss_ratio_le_0_8': ratio <= .8,
        'not_worse_than_local_control': vs_local <= 1,
        'pose_pass_count_not_decreased': int(aok.sum()) >= int(wok.sum()),
        'no_previously_qualified_probe_lost': lost == 0,
        'position_median_not_increased': float(np.median(ap)) <= float(np.median(wp)),
        'rotation_median_not_increased': float(np.median(aa)) <= float(np.median(wa)),
    }
    return {
        'scope': 'prospective TRAIN single-step gate; not a DEV/model/execution verdict',
        'median_paired_loss_ratio_vs_warm': ratio, 'median_paired_loss_ratio_vs_local': vs_local,
        'position_median_mm': float(1000*np.median(ap)),
        'rotation_median_deg': float(np.degrees(np.median(aa))),
        'max_normalized_error_p95': float(np.quantile(aw, .95)),
        'pose_pass': int(aok.sum()), 'warm_pose_pass': int(wok.sum()),
        'lost_qualified': lost, 'gained_qualified': int(np.sum(~wok & aok)),
        'case_count': len(ai), 'distinct_target_records': len(set(i[0] for i in ai)),
        'gates': gates, 'TRAIN_endpoint_gate': 'PASS' if all(gates.values()) else 'BLOCKED',
        'automatic_DEV_or_TEST_launch': False,
    }
