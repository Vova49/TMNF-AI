"""
Тестовый скрипт для прогона последней обученной модели в TMNF.

Что делает:
- Загружает чекпоинт "checkpoints/last.pt" (формат как в train.py / PPOTrainer).
- Поднимает TrackmaniaEnv (через TMInterface).
- Запускает несколько эпизодов, где машина едет по политике.
- Печатает суммарную награду и немного информации по ходу заезда.

Перед запуском:
1) Запусти TMNF + TMInterface.
2) Убедись, что плагин RL (TMInterface_RL_Plugin.txt) активен.
3) Загрузи нужную трассу.
4) Убедись, что порт и host совпадают с trackmania_rl.config.tmiface.
"""

import os
import time

import torch
from trackmania_rl.env import TrackmaniaEnv
from trackmania_rl.policy import Policy


CHECKPOINT_PATH = "checkpoints/last.pt"


def load_policy(env) -> Policy:
    """Создаёт Policy нужного размера и загружает веса из чекпоинта."""
    obs_dim = env.observation_space.shape[0]
    policy = Policy(obs_dim=obs_dim)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Чекпоинт не найден: {CHECKPOINT_PATH}\n"
            f"Сначала запусти обучение (train.py), чтобы он появился."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint.get("policy_state_dict")
    if state_dict is None:
        raise KeyError(
            "В чекпоинте нет ключа 'policy_state_dict'. "
            "Проверь, что сохранение выполнялось через PPOTrainer.save_checkpoint()."
        )

    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def run_episode(env: TrackmaniaEnv, policy: Policy, max_steps: int = 10_000) -> float:
    """Запускает один эпизод и возвращает суммарную награду."""
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0
    step_idx = 0

    with torch.no_grad():
        while not done and step_idx < max_steps:
            step_idx += 1

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            # ВАЖНО: используем ровно тот же способ семплирования, что и в train.py
            action, _ = policy.sample_action(obs_tensor)
            action_np = action.cpu().numpy()[0]

            # env.step сам отправит инпуты в TMInterface, ничего дополнительно делать не нужно
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated) or bool(truncated)
            ep_reward += float(reward)

            # Немного логов раз в N шагов, чтобы видеть прогресс
            if step_idx % 50 == 0:
                print(
                    f"[step {step_idx:5d}] "
                    f"cp_index={info.get('cp_index', -1)}, "
                    f"race_time={info.get('race_time', 0.0):7.3f}s, "
                    f"reward_step={reward:8.4f}, "
                    f"ep_reward={ep_reward:8.4f}"
                )

            # Чуть замедлим, если хочешь ближе к realtime (настрой по вкусу)
            time.sleep(0.0)

    print(
        f"Эпизод завершён: steps={step_idx}, total_reward={ep_reward:.4f}, "
        f"terminated={terminated}, truncated={truncated}"
    )
    return ep_reward


def main():
    # Создаём окружение (подключается к TMInterface и плагину)
    env = TrackmaniaEnv()

    try:
        policy = load_policy(env)
        print(f"Модель загружена из '{CHECKPOINT_PATH}'. Запускаю тестовый заезд...")

        num_episodes = 1  # можно увеличить, если хочешь несколько прогонов
        for ep in range(num_episodes):
            print(f"\n========== Эпизод {ep + 1}/{num_episodes} ==========")
            total_reward = run_episode(env, policy)
            print(f"[Эпизод {ep + 1}] total_reward = {total_reward:.4f}")

    finally:
        env.close()
        print("Окружение закрыто, соединение с TMInterface разорвано.")


if __name__ == "__main__":
    main()
