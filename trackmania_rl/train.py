"""
Цикл обучения (PPO/SAC).

Содержимое:
- Сбор rollout'ов в окружении;
- Логирование метрик;
- Сохранение/загрузка модели;
- Периодический eval.
"""
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from trackmania_rl.env import TrackmaniaEnv
from trackmania_rl.policy import Policy

warnings.filterwarnings("ignore", category=UserWarning)


class RolloutBuffer:
    """Буфер для хранения траекторий."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.brake_probs = []
        self.dones = []

    def add(self, obs, action, reward, log_prob, brake_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.brake_probs.append(brake_prob)
        self.dones.append(done)

    def get(self):
        return (
            np.array(self.obs),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.log_probs),
            np.array(self.brake_probs),
            np.array(self.dones)
        )

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.brake_probs.clear()
        self.dones.clear()


class PPODataset(Dataset):
    """Dataset для батчевого обучения PPO."""

    def __init__(self, obs, actions, advantages, returns, old_log_probs, old_brake_probs):
        self.obs = obs
        self.actions = actions
        self.advantages = advantages
        self.returns = returns
        self.old_log_probs = old_log_probs
        self.old_brake_probs = old_brake_probs

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32),
            torch.as_tensor(self.actions[idx], dtype=torch.float32),
            torch.as_tensor(self.advantages[idx], dtype=torch.float32),
            torch.as_tensor(self.returns[idx], dtype=torch.float32),
            torch.as_tensor(self.old_log_probs[idx], dtype=torch.float32),
            torch.as_tensor(self.old_brake_probs[idx], dtype=torch.float32)
        )


class PPOTrainer:
    """Реализация Proximal Policy Optimization."""

    def __init__(
            self,
            env: TrackmaniaEnv,
            policy: Policy,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            value_loss_coef: float = 0.5,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            num_epochs: int = 10,
            batch_size: int = 64,
    ):
        self.env = env
        self.policy = policy.to(device)
        self.device = device

        # Оптимизатор
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

        # Гиперпараметры
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Value function
        self.value_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Буфер для сбора данных
        self.buffer = RolloutBuffer()

        # Bootstrap storage for GAE (V(s_T) when last step is non-terminal)
        self._bootstrap_value = 0.0
        self._bootstrap_done = 1.0

    def collect_rollout(self, num_steps: int):
        """Сбор траектории фиксированной длины."""
        obs, _ = self.env.reset()
        last_next_obs = obs
        last_done = True

        for _ in range(num_steps):
            # Получаем действие от политики
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, info = self.policy.sample_action(obs_tensor)
                action = action.cpu().numpy()[0]

            # Делаем шаг в окружении (Gymnasium API)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated) or bool(truncated)

            # Сохраняем в буфер
            self.buffer.add(
                obs, action, reward,
                info['log_prob'].cpu().numpy()[0],
                info['brake_prob'].cpu().numpy()[0],
                done
            )

            # Трекинг последнего next_obs и done для бутстрапа
            last_next_obs = next_obs
            last_done = done

            # Переход к следующему состоянию/эпизоду
            if done:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        # Сохраняем bootstrap value для GAE
        with torch.no_grad():
            if not last_done:
                obs_tensor = torch.as_tensor(last_next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                self._bootstrap_value = self.value_net(obs_tensor).squeeze().item()
                self._bootstrap_done = 0.0
            else:
                self._bootstrap_value = 0.0
                self._bootstrap_done = 1.0


    def compute_advantages(self):
        """Вычисление advantage function и returns."""
        obs = torch.FloatTensor(self.buffer.obs).to(self.device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        dones = torch.FloatTensor(self.buffer.dones).to(self.device)

        with torch.no_grad():
            values = self.value_net(obs).squeeze()
            # next_values: V(s_{t+1}); для последнего шага используем bootstrap V(s_T) если он нетерминальный
            shifted = torch.cat([values[1:], values[-1:].clone()])
            last_nv = torch.as_tensor(self._bootstrap_value, device=self.device, dtype=shifted.dtype)
            # если последний шаг терминальный, множитель (1 - dones[-1]) занулит bootstrap
            shifted[-1] = last_nv

        # GAE
        advantages = []
        gae = 0.0
        T = len(rewards)
        for t in reversed(range(T)):
            next_v = shifted[t]
            delta = rewards[t] + self.gamma * next_v * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.as_tensor(advantages, device=self.device, dtype=values.dtype)
        returns = advantages + values

        # Нормализация advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.cpu().numpy(), returns.cpu().numpy()


    def update(self):
        """Обновление политики и value function."""
        # Получаем данные из буфера
        obs, actions, _, old_log_probs, old_brake_probs, _ = self.buffer.get()
        advantages, returns = self.compute_advantages()

        # Создаем dataset для батчевого обучения
        dataset = PPODataset(
            obs, actions, advantages, returns,
            old_log_probs, old_brake_probs
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Обучение
        for _ in range(self.num_epochs):
            for batch in dataloader:
                (
                    b_obs, b_actions, b_advantages, b_returns,
                    b_old_log_probs, b_old_brake_probs
                ) = [x.to(self.device) for x in batch]

                # Forward pass
                normal_dist, brake_probs = self.policy(b_obs)
                values = self.value_net(b_obs).squeeze()

                # Continuous actions (steer, throttle)
                # b_actions contains throttle in [0,1]; convert back to [-1,1] for log-prob under Normal
                steer = b_actions[:, 0]
                throttle01 = b_actions[:, 1]
                throttle = torch.clamp(throttle01 * 2.0 - 1.0, -1.0, 1.0)
                continuous_actions_pre = torch.stack([steer, throttle], dim=-1)
                log_probs = normal_dist.log_prob(continuous_actions_pre).sum(-1)

                # Binary action (brake) – текущие и старые log_probs
                brake_actions = b_actions[:, 2]
                brake_log_probs = (
                        brake_actions * torch.log(brake_probs + 1e-8) +
                        (1 - brake_actions) * torch.log(1 - brake_probs + 1e-8)
                ).squeeze()
                old_brake_log_probs = (
                        brake_actions * torch.log(b_old_brake_probs + 1e-8) +
                        (1 - brake_actions) * torch.log(1 - b_old_brake_probs + 1e-8)
                ).squeeze()

                # Совокупный логарифм вероятности для PPO
                total_log_probs = log_probs + brake_log_probs
                old_total_log_probs = b_old_log_probs + old_brake_log_probs

                # Ratio для PPO
                ratio = torch.exp(total_log_probs - old_total_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = self.value_loss_coef * nn.MSELoss()(values, b_returns)

                # Entropy bonus
                entropy = normal_dist.entropy().mean()
                brake_entropy = -(
                        brake_probs * torch.log(brake_probs + 1e-8) +
                        (1 - brake_probs) * torch.log(1 - brake_probs + 1e-8)
                ).mean()
                entropy_loss = -self.entropy_coef * (entropy + brake_entropy)

                # Полный лосс
                loss = policy_loss + value_loss + entropy_loss

                # Backprop
                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) +
                    list(self.value_net.parameters()),
                    self.max_grad_norm
                )

                self.optimizer.step()
                self.value_optimizer.step()

        # Очищаем буфер
        self.buffer.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

    def save_checkpoint(self, path: str):
        """Сохранение чекпоинта."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, path)

    def load_checkpoint(self, path: str):
        """Загрузка чекпоинта."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])


def main():
    """Основной цикл обучения."""

    # Создание окружения
    env = TrackmaniaEnv()

    # (опционально) Принудительный респавн до старта обучения,
    # чтобы первый заезд точно шёл с нуля
    obs, _ = env.reset()
    print("[train] initial reset done")

    # Создание политики
    policy = Policy(obs_dim=env.observation_space.shape[0])

    # Создание тренера
    trainer = PPOTrainer(env, policy)

    # Попытка загрузить последнюю сохранённую модель
    checkpoint_path = "checkpoints/last.pt"
    if os.path.exists(checkpoint_path):
        try:
            trainer.load_checkpoint(checkpoint_path)
            print(f"Модель успешно загружена из {checkpoint_path}")
        except Exception as e:
            print(f"Не удалось загрузить чекпоинт ({e}), обучение начнётся с нуля.")
    else:
        print("Чекпоинт не найден, обучение начнётся с нуля.")

    # Параметры обучения
    num_iterations = 1000  # Количество итераций
    steps_per_iteration = 2048  # Шагов за итерацию
    eval_freq = 10  # Частота оценки
    save_freq = 5  # Частота сохранения

    for iteration in range(num_iterations):
        print(f"Iteration {iteration}")

        # Сбор данных
        trainer.collect_rollout(steps_per_iteration)

        # Обновление политики
        metrics = trainer.update()

        # Сохранение чекпоинта
        if (iteration + 1) % save_freq == 0:
            trainer.save_checkpoint("checkpoints/last.pt")

        # Оценка политики
        if (iteration + 1) % eval_freq == 0:
            with torch.no_grad():
                eval_reward = 0
                obs, _ = env.reset()
                done = False

                while not done:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
                    action, _ = trainer.policy.sample_action(obs_tensor)
                    obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
                    done = bool(terminated) or bool(truncated)
                    eval_reward += reward


if __name__ == "__main__":
    main()
