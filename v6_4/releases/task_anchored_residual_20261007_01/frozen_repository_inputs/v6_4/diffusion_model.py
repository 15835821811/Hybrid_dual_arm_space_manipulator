"""Conditional DDPM for free B-spline control points, with explicit loss identity.

The model proposes 30 x 17 free coordinates.  It never fixes boundaries, clips
proposals, changes task targets, produces torques, or invokes a controller.
Boundary elimination and trajectory decoding belong to trajectory_codec.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DiffusionConfig:
    condition_dim: int
    free_control_points: int = 30
    planner_dimensions: int = 17
    hidden_dim: int = 128
    layers: int = 4
    heads: int = 4
    diffusion_steps: int = 100
    ddim_steps: int = 20
    noise_schedule: str = "cosine"
    parameterization: str = "epsilon_residual"
    objective: str = "epsilon_mse"
    cosine_s: float = .008
    beta_cap: float = .999
    beta_start: float = 1e-4
    beta_end: float = .02

    def __post_init__(self):
        if self.condition_dim < 1:
            raise ValueError("condition_dim must be positive")
        if (self.free_control_points, self.planner_dimensions, self.hidden_dim,
                self.layers, self.heads, self.diffusion_steps) != (30, 17, 128, 4, 4, 100):
            raise ValueError("V6.4-A architecture is fixed at 30x17, 4x128/4 heads, DDPM100")
        if not 1 <= self.ddim_steps <= self.diffusion_steps:
            raise ValueError("invalid DDIM inference step count")
        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("invalid DDPM beta schedule")
        if self.noise_schedule not in ("cosine", "linear_legacy_unscaled"):
            raise ValueError("unknown noise schedule")
        if self.parameterization not in ("epsilon_residual", "direct_epsilon", "clean_x0"):
            raise ValueError("unknown epsilon parameterization")
        if self.objective not in ("epsilon_mse", "v_mse"):
            raise ValueError("unknown training objective")
        if self.objective == "v_mse" and self.parameterization not in ("epsilon_residual", "clean_x0"):
            raise ValueError("v_mse requires an explicit residual or clean_x0 parameterization")
        if self.parameterization == "clean_x0" and self.objective != "v_mse":
            raise ValueError("this clean_x0 version fixes its objective to v_mse")
        if self.cosine_s != .008 or self.beta_cap != .999:
            raise ValueError("V6.4-A cosine100 fixes s=.008 and beta_cap=.999")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value, *, allow_legacy_missing_schedule=False, allow_legacy_missing_parameterization=False,
                  allow_legacy_missing_objective=False):
        config = dict(value)
        if "noise_schedule" not in config:
            if not allow_legacy_missing_schedule:
                raise ValueError("model config must explicitly identify its noise schedule")
            # v1 smoke used these unscaled 100-step linear endpoints. Never
            # reinterpret its existing weights/buffers as cosine-trained.
            config["noise_schedule"] = "linear_legacy_unscaled"
        if "parameterization" not in config:
            if not allow_legacy_missing_parameterization:
                raise ValueError("model config must explicitly identify its epsilon parameterization")
            config["parameterization"] = "direct_epsilon"
        if "objective" not in config:
            if not allow_legacy_missing_objective:
                raise ValueError("model config must explicitly identify its training objective")
            config["objective"] = "epsilon_mse"
        return cls(**config)


def parameterization_identity(config: DiffusionConfig):
    if config.parameterization == "clean_x0":
        value = {"schema": "v6_4_clean_x0_parameterization_v1", "parameterization": "clean_x0",
                 "network_output": "unclipped_clean_x0", "epsilon_formula": "(xt-sqrt(alpha_bar)*x0_net)/sqrt(1-alpha_bar)",
                 "x0_formula": "network_output", "loss_target": "true_forward_v",
                 "loss": "mean((x0_net-x0_true)^2/(1-alpha_bar))",
                 "equivalent_internal_clean_mse_weight": "1/(1-alpha_bar(t))",
                 "v_formula": "(sqrt(alpha_bar)*xt-x0_net)/sqrt(1-alpha_bar)",
                 "terminal_risk": "inverse noise coefficient is bounded by the fixed cosine100 schedule; no tiny-signal division",
                 "postprocessing_clipping_projection_or_reference_optimization": False,
                 "requires_parameterization_specific_training": True}
        value["parameterization_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return value
    residual = config.parameterization == "epsilon_residual"
    value = {"schema": "v6_4_epsilon_parameterization_v1", "parameterization": config.parameterization,
             "network_output": "internal_v_residual" if residual else "epsilon",
             "epsilon_formula": "sqrt(1-alpha_bar)*xt+sqrt(alpha_bar)*r_net" if residual else "network_output",
             "x0_formula": "sqrt(alpha_bar)*xt-sqrt(1-alpha_bar)*r_net" if residual else "(xt-sqrt(1-alpha_bar)*epsilon)/sqrt(alpha_bar)",
             "loss_target": "true_forward_epsilon", "loss": "basic_epsilon_mean_squared_error",
             "equivalent_internal_residual_mse_weight": "alpha_bar(t)" if residual else None,
             "terminal_residual_supervision_weight": "terminal_alpha_bar" if residual else None,
             "terminal_risk": "epsilon MSE weakly supervises the residual at low alpha; monitor x0 and epsilon_MSE/alpha separately" if residual else "epsilon error is amplified by sqrt((1-alpha)/alpha) during x0 reconstruction",
             "postprocessing_clipping_projection_or_reference_optimization": False,
             "requires_parameterization_specific_training": True}
    if config.objective == "v_mse":
        value.update(schema="v6_4_v_parameterization_v2", loss_target="true_forward_v",
                     loss="basic_v_mean_squared_error", equivalent_internal_residual_mse_weight="1",
                     terminal_residual_supervision_weight="1",
                     terminal_risk="v MSE changes timestep weighting; finite fit does not establish task or physical success")
    value["parameterization_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


def objective_identity(config: DiffusionConfig):
    value = {"schema": "v6_4_explicit_training_objective_v1", "objective": config.objective,
             "target": "sqrt(alpha_bar)*epsilon-sqrt(1-alpha_bar)*x0" if config.objective == "v_mse" else "true_forward_epsilon",
             "loss": "mean_squared_error", "additional_losses": [],
             "sampling_prediction": "epsilon_and_unclipped_x0",
             "residual_error_relation": "L_epsilon_per_example=alpha_bar*L_v;L_x0_per_example=(1-alpha_bar)*L_v",
             "epsilon_timestep_weight_of_v_mse": "1/alpha_bar" if config.objective == "v_mse" else "1",
             "is_basic_epsilon_mse": config.objective == "epsilon_mse"}
    value["objective_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


def build_noise_schedule(config: DiffusionConfig):
    """Float64 discretization, then the exact float32 buffers used by DDPM.

    Cosine follows Nichol/Dhariwal's released betas_for_alpha_bar with beta<1
    to keep the epsilon-based DDIM x0 reconstruction nonsingular.
    """
    if config.noise_schedule == "linear_legacy_unscaled":
        betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps, dtype=torch.float64)
    else:
        time = torch.arange(config.diffusion_steps + 1, dtype=torch.float64) / config.diffusion_steps
        alpha_curve = torch.cos((time + config.cosine_s) / (1 + config.cosine_s) * math.pi / 2).square()
        betas = (1 - alpha_curve[1:] / alpha_curve[:-1]).clamp(max=config.beta_cap)
    alpha_bars = torch.cumprod(1 - betas, dim=0)
    if torch.any(betas <= 0) or torch.any(betas >= 1) or torch.any(alpha_bars <= 0):
        raise ValueError("epsilon DDIM requires positive nonsingular noise schedule")
    return betas.to(torch.float32), alpha_bars.to(torch.float32)


def noise_schedule_identity(config: DiffusionConfig, betas=None, alpha_bars=None):
    if betas is None or alpha_bars is None:
        betas, alpha_bars = build_noise_schedule(config)
    beta_bytes = betas.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
    alpha_bytes = alpha_bars.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
    terminal = float(alpha_bars[-1])
    return {"schema": "v6_4_ddpm_effective_f32_schedule_v1", "noise_schedule": config.noise_schedule,
            "diffusion_steps": config.diffusion_steps, "prediction_type": "epsilon",
            "cosine_s": config.cosine_s if config.noise_schedule == "cosine" else None,
            "beta_cap": config.beta_cap if config.noise_schedule == "cosine" else None,
            "linear_beta_start": config.beta_start if config.noise_schedule != "cosine" else None,
            "linear_beta_end": config.beta_end if config.noise_schedule != "cosine" else None,
            "betas_sha256": hashlib.sha256(beta_bytes).hexdigest(),
            "alpha_bars_sha256": hashlib.sha256(alpha_bytes).hexdigest(),
            "schedule_sha256": hashlib.sha256(b"v6_4_ddpm_effective_f32_schedule_v1\0" + beta_bytes + alpha_bytes).hexdigest(),
            "hash_encoding": "schema+NUL+100 little-endian float32 betas+100 little-endian float32 alpha_bars",
            "terminal_alpha_bar": terminal, "terminal_signal_coefficient": math.sqrt(terminal),
            "terminal_snr": terminal / (1 - terminal), "terminal_alpha_exact_zero": False,
            "buffer_dtype": "float32", "schedule_calculation": "float64 betas/cumprod then float32"}


def timestep_embedding(timesteps: torch.Tensor, width: int = 128) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(-math.log(10000.) * torch.arange(half, device=timesteps.device,
                                                          dtype=torch.float32) / max(half - 1, 1))
    angles = timesteps.to(torch.float32)[:, None] * frequencies[None]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    return F.pad(embedding, (0, width - embedding.shape[-1]))


class ConditionalDenoiser(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        width = config.hidden_dim
        self.control_input = nn.Linear(config.planner_dimensions, width)
        self.condition_input = nn.Sequential(nn.Linear(config.condition_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time_input = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.randn(1, config.free_control_points + 1, width) * .02)
        layer = nn.TransformerEncoderLayer(width, config.heads, 4 * width, dropout=0.,
                                          activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.layers, norm=nn.LayerNorm(width),
                                                 enable_nested_tensor=False)
        # PyTorch clones an encoder template; initialize each layer independently.
        for block in self.transformer.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.output = nn.Linear(width, config.planner_dimensions)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        batch = noisy.shape[0]
        if noisy.shape != (batch, 30, 17) or condition.shape != (batch, self.config.condition_dim):
            raise ValueError("expected noisy [B,30,17] and condition [B,condition_dim]")
        if timesteps.shape != (batch,) or timesteps.dtype not in (torch.int32, torch.int64):
            raise ValueError("timesteps must be an integer vector")
        if torch.any((timesteps < 0) | (timesteps >= self.config.diffusion_steps)):
            raise ValueError("DDPM timestep outside declared schedule")
        temporal = self.time_input(timestep_embedding(timesteps, self.config.hidden_dim))[:, None]
        controls = self.control_input(noisy) + temporal
        context = self.condition_input(condition)[:, None] + temporal
        encoded = self.transformer(torch.cat((context, controls), dim=1) + self.position)
        return self.output(encoded[:, 1:])


class ConditionalDDPM(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.denoiser = ConditionalDenoiser(config)
        betas, alpha_bars = build_noise_schedule(config)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    def schedule_identity(self):
        return noise_schedule_identity(self.config, self.betas, self.alpha_bars)

    def parameterization_identity(self):
        return parameterization_identity(self.config)

    def _coefficients(self, timesteps):
        alpha = self.alpha_bars[timesteps].reshape(-1, 1, 1)
        return alpha.sqrt(), (1 - alpha).sqrt()

    def forward(self, noisy, timesteps, condition):
        raw = self.denoiser(noisy, timesteps, condition)
        if self.config.parameterization == "direct_epsilon":
            return raw
        signal, noise = self._coefficients(timesteps)
        if self.config.parameterization == "clean_x0":
            return (noisy - signal * raw) / noise
        return noise * noisy + signal * raw

    def predict_epsilon_and_x0(self, noisy, timesteps, condition):
        raw = self.denoiser(noisy, timesteps, condition)
        signal, noise = self._coefficients(timesteps)
        if self.config.parameterization == "direct_epsilon":
            return raw, (noisy - noise * raw) / signal
        if self.config.parameterization == "clean_x0":
            # The raw network output IS clean x0. No proposal exists yet, and
            # neither task information nor a clamp/projection modifies it.
            return (noisy - signal * raw) / noise, raw
        # A fixed internal coordinate change, before a raw proposal exists.
        # The config separately identifies epsilon MSE or v MSE training. The
        # prediction coordinate change is algebraically the same
        # DDIM x0, without subtracting nearly equal epsilon/xt and dividing by
        # tiny sqrt(alpha). No bounds or task information is applied here.
        epsilon = noise * noisy + signal * raw
        return epsilon, signal * noisy - noise * raw

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor,
                 noise: torch.Tensor | None = None, generator: torch.Generator | None = None):
        if noise is None:
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        if noise.shape != clean.shape:
            raise ValueError("epsilon shape differs from clean controls")
        alpha = self.alpha_bars[timesteps].reshape(-1, 1, 1)
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise, noise

    def epsilon_loss(self, clean, condition, timesteps=None, noise=None, generator=None):
        if timesteps is None:
            timesteps = torch.randint(0, self.config.diffusion_steps, (clean.shape[0],),
                                      device=clean.device, generator=generator)
        noisy, epsilon = self.q_sample(clean, timesteps, noise, generator)
        return F.mse_loss(self(noisy, timesteps, condition), epsilon)

    def v_loss(self, clean, condition, timesteps=None, noise=None, generator=None):
        if self.config.parameterization not in ("epsilon_residual", "clean_x0"):
            raise ValueError("v loss cannot reinterpret a direct epsilon network")
        if timesteps is None:
            timesteps = torch.randint(0, self.config.diffusion_steps, (clean.shape[0],),
                                      device=clean.device, generator=generator)
        noisy, epsilon = self.q_sample(clean, timesteps, noise, generator)
        signal, noise_coefficient = self._coefficients(timesteps)
        if self.config.parameterization == "clean_x0":
            predicted_clean = self.denoiser(noisy, timesteps, condition)
            # v_pred-v_true = -(x0_pred-x0_true)/sqrt(1-alpha_bar).
            # Per-example weighting preserves the SAME v-MSE objective;
            # averaging first and dividing by mean noise variance would not.
            return ((predicted_clean - clean).square() / noise_coefficient.square()).mean()
        target = signal * epsilon - noise_coefficient * clean
        return F.mse_loss(self.denoiser(noisy, timesteps, condition), target)

    def training_loss(self, clean, condition, timesteps=None, noise=None, generator=None):
        loss = self.v_loss if self.config.objective == "v_mse" else self.epsilon_loss
        return loss(clean, condition, timesteps, noise, generator)

    @torch.no_grad()
    def sample_ddim(self, condition: torch.Tensor, *, steps: int = 20,
                    generator: torch.Generator | None = None,
                    initial_noise: torch.Tensor | None = None) -> torch.Tensor:
        if not 1 <= steps <= self.config.diffusion_steps:
            raise ValueError("invalid DDIM step count")
        if condition.ndim != 2 or condition.shape[1] != self.config.condition_dim:
            raise ValueError("invalid condition batch")
        shape = (condition.shape[0], 30, 17)
        if initial_noise is None:
            sample = torch.randn(shape, device=condition.device, dtype=condition.dtype, generator=generator)
        else:
            if initial_noise.shape != shape:
                raise ValueError("initial noise shape mismatch")
            sample = initial_noise.clone()
        times = torch.linspace(self.config.diffusion_steps - 1, 0, steps, device=condition.device).round().long()
        was_training = self.training
        self.eval()
        try:
            for index, time in enumerate(times):
                t = torch.full((shape[0],), int(time), dtype=torch.long, device=condition.device)
                alpha = self.alpha_bars[time]
                previous = self.alpha_bars[times[index + 1]] if index + 1 < len(times) else sample.new_tensor(1.)
                epsilon, clean = self.predict_epsilon_and_x0(sample, t, condition)
                # Deterministic DDIM (eta=0).  Raw x0 remains unclipped.
                sample = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
            return sample
        finally:
            self.train(was_training)
