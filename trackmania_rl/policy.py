"""Политика/модель управления.

Назначение:
- Преобразование наблюдений в действие a = (steer, throttle, brake);
- Маппинг action_to_inputs(a) для отправки в TMInterface.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):
    """Политика на основе нейронной сети для управления в TrackMania.
    
    Архитектура:
    - Наблюдения -> FC слои -> (μ, σ) для каждого действия
    - Действия семплируются из Normal(μ, σ)
    - tanh для ограничения в [-1, 1]
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # μ и σ для steer и throttle
        self.mean = nn.Linear(hidden_dim, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

        # Отдельная head для binary brake
        self.brake_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """Forward pass политики.
        
        Args:
            obs: Тензор наблюдений [batch_size, obs_dim]
            
        Returns:
            tuple[Normal, torch.Tensor]: 
                - Распределение Normal(μ, σ) для steer и throttle
                - Вероятности brake (sigmoid)
        """
        features = self.net(obs)

        mean = self.mean(features)
        std = self.log_std.exp()
        normal_dist = Normal(mean, std)

        brake_logits = self.brake_head(features)
        brake_probs = torch.sigmoid(brake_logits)

        return normal_dist, brake_probs

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Семплирование действия из политики.
        
        Args:
            obs: Тензор наблюдений [batch_size, obs_dim]
            
        Returns:
            tuple[torch.Tensor, dict]:
                - Действие [batch_size, 3] (steer, throttle, brake)
                - Дополнительная информация для обучения
        """
        normal_dist, brake_probs = self.forward(obs)
        z = normal_dist.rsample()
        a = torch.tanh(z)
        brake = torch.bernoulli(brake_probs)
        steer = a[:, 0].unsqueeze(-1)
        throttle01 = torch.clamp(0.5 * (a[:, 1] + 1.0), 0.0, 1.0).unsqueeze(-1)
        action = torch.cat([steer, throttle01, brake], dim=-1)
        log_det = torch.log(1.0 - a.pow(2) + 1e-6).sum(-1)
        base_log_prob = normal_dist.log_prob(z).sum(-1)
        info = {
            'log_prob': base_log_prob - log_det,
            'brake_prob': brake_probs
        }
        return action, info

    @staticmethod
    def action_to_inputs(action: np.ndarray) -> tuple[float, float, bool, bool]:
        """Преобразование действий в формат TMInterface.
        
        Args:
            action: Массив [steer, throttle, brake]
            
        Returns:
            tuple[float, float, bool, bool]: (steer, gas, brake, handbrake)
                - steer: [-1, 1]
                - gas: [0, 1]
                - brake: bool
                - handbrake: всегда False
        """
        steer = float(np.clip(action[0], -1, 1))
        throttle = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))  # [-1,1] -> [0,1]
        brake = bool(action[2] > 0.5)
        handbrake = False  # Пока не используем

        return steer, throttle, brake, handbrake

    def log_prob_squashed(self, normal_dist: Normal, squashed_actions: torch.Tensor) -> torch.Tensor:
        z = torch.atanh(torch.clamp(squashed_actions, -0.999999, 0.999999))
        base = normal_dist.log_prob(z).sum(-1)
        log_det = torch.log(1.0 - squashed_actions.pow(2) + 1e-6).sum(-1)
        return base - log_det
