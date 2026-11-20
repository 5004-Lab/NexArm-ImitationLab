"""Policies share LeRobot features, processors, checkpoints and action API."""

from collections import deque
from dataclasses import dataclass, field

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@PreTrainedConfig.register_subclass("bc_mlp")
@dataclass
class BCConfig(PreTrainedConfig):
    chunk_size: int = 1
    n_action_steps: int = 1
    hidden_dim: int = 256
    pretrained_backbone_weights: str | None = None
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    normalization_mapping: dict = field(
        default_factory=lambda: {
            k: NormalizationMode.MEAN_STD for k in ("VISUAL", "STATE", "ACTION")
        }
    )

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        pass


class BCPolicy(PreTrainedPolicy):
    config_class = BCConfig
    name = "bc_mlp"

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        weights = ResNet18_Weights.IMAGENET1K_V1 if config.pretrained_backbone_weights else None
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()
        size = config.input_features["observation.state"].shape[0] + 512 * len(
            config.image_features
        )
        action_dim = config.output_features["action"].shape[0]
        self.head = nn.Sequential(
            nn.Linear(size, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.chunk_size * action_dim),
        )
        self.reset()

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        self.queue = deque()

    def _predict(self, batch):
        features = [batch["observation.state"]] + [
            self.encoder(batch[k]) for k in self.config.image_features
        ]
        result = self.head(torch.cat(features, dim=-1))
        return result.reshape(
            -1, self.config.chunk_size, self.config.output_features["action"].shape[0]
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        self.eval()
        return self._predict(batch)

    @torch.no_grad()
    def select_action(self, batch):
        if not self.queue:
            self.queue.extend(
                self.predict_action_chunk(batch)[:, : self.config.n_action_steps].transpose(0, 1)
            )
        return self.queue.popleft()

    def forward(self, batch):
        error = (self._predict(batch) - batch["action"]).abs()
        mask = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(error)
        loss = error[mask].mean()
        return loss, {"l1_loss": loss.item()}


def make_policy(cfg, meta):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    features = {}
    for name, feature in meta.features.items():
        if name == "observation.state":
            features[name] = PolicyFeature(type=FeatureType.STATE, shape=tuple(feature["shape"]))
        elif name.startswith("observation.images."):
            h, w, c = feature["shape"]
            features[name] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
    options = dict(cfg["policy"])
    kind = options.pop("type", "act")
    config_class, policy_class = {
        "act": (ACTConfig, ACTPolicy),
        "bc_mlp": (BCConfig, BCPolicy),
        "diffusion": (DiffusionConfig, DiffusionPolicy),
    }[kind]
    config = config_class(
        input_features=features,
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION, shape=tuple(meta.features["action"]["shape"])
            )
        },
        device=cfg["device"],
        push_to_hub=False,
        **options,
    )
    return policy_class(config).to(cfg["device"])


def processors(config, stats):
    if config.type == "diffusion":
        from lerobot.policies.diffusion.processor_diffusion import (
            make_diffusion_pre_post_processors,
        )

        return make_diffusion_pre_post_processors(config, stats)
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    return make_act_pre_post_processors(config, stats)


def validation_prediction(policy, batch):
    if policy.config.type == "diffusion":
        batch = dict(batch)
        batch["observation.images"] = torch.stack(
            [batch[k] for k in policy.config.image_features], dim=-4
        )
        return policy.diffusion.generate_actions(batch)[:, 0]
    return policy.predict_action_chunk(batch)[:, 0]
