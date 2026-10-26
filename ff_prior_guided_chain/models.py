"""Inference-only loader for the existing aligned checkpoint's own metadata schema."""
import torch
from ff_prior_latent.learning import Codec,LatentDenoiser
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.diffusion import weight_hash


def load_static(directory):
    report=read_json(directory/'training.json')
    saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    assert file_digest(directory/'model.pt')==report['checkpoint_sha256']
    assert saved['metadata']==report['metadata']
    codec=Codec(read_json(directory/'codec.json'))
    assert digest(codec.state)==report['metadata']['codec_hash']
    model=LatentDenoiser(102,65);model.load_state_dict(saved['weights']);model.eval()
    assert weight_hash(model)==report['trained_weight_hash']
    for p in model.parameters():p.requires_grad_(False)
    return model,codec
