"""C4-A explicit, byte-verified relocation of frozen C3 inference artifacts.

No training, selection, repair, or physics. The published C3 teacher has a
documented post-experiment metadata fix; that module is not an inference input.
All other model-production sources must match the frozen bytes exactly.
"""
from pathlib import Path
import time

import torch

from .route_optimizer_protocol import read, sha
from .dataset import object_sha
from .residual_diffusion import ResidualDDPM, ResidualDiffusionConfig, _configure, _state_sha
from .simple_warmstart_regression import SearchAwareSampler, SimpleWarmstartRegression, CHECKPOINT_SCHEMA
from .preference_teacher_dataset import ConditionNormalizer
from .residual_dataset import ResidualNormalizer
from .visualization.export_search_aware_release import PortableResolver

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/'v6_4/releases/search_aware_warmstart_20261008_01'


def load_frozen_sampler(name='D', release=RELEASE):
    started = time.perf_counter()
    resolver = PortableResolver(release)
    freeze = read(resolver.snapshot('model_freeze.json'))
    for original, expected in freeze['artifacts'].items():
        resolver.resolve(original, expected)
    path = resolver.resolve(freeze['selected_checkpoints'][name])
    c = torch.load(path, map_location='cpu', weights_only=False)
    if c['schema'] != CHECKPOINT_SCHEMA or c['model_name'] != name or c['update'] != 4000:
        raise ValueError('C4-A only uses pre-TEST selected D4000/S4000')
    checks = {}
    for original, expected in c['training_config']['source_files'].items():
        resolver.resolve(original, expected)
        relative = 'v6_4/' + Path(original).name
        if relative == 'v6_4/search_effect_teacher.py':
            checks[relative] = {'frozen_verified': True, 'role': 'training producer only; not called by inference'}
        else:
            if sha(ROOT/relative) != expected:
                raise ValueError('C4-A inference dependency differs: '+relative)
            checks[relative] = {'frozen_verified': True, 'loaded_source_verified': True}
    if sha(resolver.snapshot('models/training_config.json')) != c['training_config_sha256']:
        raise ValueError('training config differs')
    resolver.resolve(c['training_config']['dataset_manifest_path'], c['training_config']['dataset_manifest_sha256'])
    _configure()
    s = SearchAwareSampler.__new__(SearchAwareSampler)
    s.checkpoint_path, s.checkpoint_sha256 = path, sha(path)
    s.device, s.model_name = torch.device('cpu'), name
    s.model = (ResidualDDPM(ResidualDiffusionConfig.from_dict(c['model_config'])) if name == 'D'
               else SimpleWarmstartRegression(c['model_config']['condition_dim']))
    s.model.load_state_dict(c['state_dict'], strict=True)
    s.model.eval()
    if _state_sha(s.model) != c['state_sha256']:
        raise ValueError('checkpoint tensor identity differs')
    for parameter in s.model.parameters():
        parameter.requires_grad_(False)
    s.checkpoint = c
    s.condition_normalizer = ConditionNormalizer.from_dict(c['condition_normalizer'])
    s.residual_normalizer = ResidualNormalizer.from_dict(c['residual_normalizer'])
    s.identity = dict(checkpoint_sha256=s.checkpoint_sha256, checkpoint_update=c['update'], model=name,
        training_config_sha256=c['training_config_sha256'], condition_scaler_sha256=object_sha(c['condition_normalizer']),
        residual_scaler_sha256=object_sha(c['residual_normalizer']), condition_schema_sha256=object_sha(c['condition_schema']),
        effective_parameter_count=c['effective_parameter_count'])
    s.model_load_wall_s = time.perf_counter()-started
    s.sample_units = 0
    s.c4a_source_checks = checks
    return s
