"""Read-only playback adapter for the copied 48-input flat CRA373 PPO actor.

The rough direct environment inserts 187 height samples before its final 12
previous-action values. The flat actor was trained on the other 48 values in
that order. This adapter restores that input layout without changing the model,
training code, environment observation or action definitions.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


def flat_observation_from_rough(rough_policy_obs: torch.Tensor) -> torch.Tensor:
    if rough_policy_obs.ndim != 2 or rough_policy_obs.shape[1] != 235:
        raise ValueError("CRA373 rough policy observation must have shape (N, 235)")
    return torch.cat((rough_policy_obs[:, :36], rough_policy_obs[:, -12:]), dim=1)


class FlatPPOPolicy:
    def __init__(self, checkpoint: str | Path, device: str | torch.device):
        path = Path(checkpoint)
        state = torch.load(path, map_location=device, weights_only=True)["actor_state_dict"]
        # Exact architecture from the copied flat run's params/agent.yaml.
        self.model = nn.Sequential(
            nn.Linear(48, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12),
        ).to(device)
        self.model.load_state_dict({key.removeprefix("mlp."): value
                                    for key, value in state.items() if key.startswith("mlp.")})
        self.model.eval()
        self.checkpoint = path

    @torch.inference_mode()
    def __call__(self, rough_policy_obs: torch.Tensor) -> torch.Tensor:
        return self.model(flat_observation_from_rough(rough_policy_obs))
