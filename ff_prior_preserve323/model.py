from ff_prior_control_transfer.model import TransferDenoiser as Original,rollout,teacher_loss,initial_noise
from ff_prior.diffusion import weight_hash

class TransferDenoiser(Original):
    def __init__(self,codec,warm=True):
        super().__init__(codec,warm=warm)
        self.low.requires_grad_(False)

    def trainable(self):
        return list(self.detail.parameters())

    def low_hash(self):
        return weight_hash(self.low)
