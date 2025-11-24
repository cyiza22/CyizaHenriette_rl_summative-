import os
import sys
import numpy as np
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import BreastCancerAwarenessEnv


class MetricsCallback(BaseCallback):
    """Callback for tracking training metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_women_reached = []
        self.episode_coverage_rates = []
        
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
            self.episode_women_reached.append(info.get('women_reached', 0))
            self.episode_coverage_rates.append(info.get('coverage_rate', 0))
        return True


def train_ppo(
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    total_timesteps=200000,
    run_name="ppo_run"
):
    """Train PPO agent."""
    
    env = BreastCancerAwarenessEnv(render_mode=None)
    env = Monitor(env)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        policy_kwargs=dict(net_arch=[256, 256]),  # Bigger network
        verbose=1,
        tensorboard_log=f"./logs/ppo/{run_name}"
    )
    
    callback = MetricsCallback()
    
    print(f"\nTraining PPO with configuration:")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"N Steps: {n_steps}")
    print(f"Batch Size: {batch_size}")
    print(f"N Epochs: {n_epochs}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
        progress_bar=True
    )
    
    os.makedirs("models/ppo", exist_ok=True)
    model.save(f"models/ppo/{run_name}")
    
    mean_reward = np.mean(callback.episode_rewards[-100:]) if len(callback.episode_rewards) > 0 else 0
    mean_coverage = np.mean(callback.episode_coverage_rates[-100:]) if len(callback.episode_coverage_rates) > 0 else 0
    
    # Save metrics
    metrics = {
        'learning_rate': learning_rate,
        'gamma': gamma,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'clip_range': clip_range,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'total_timesteps': total_timesteps,
        'mean_reward': float(mean_reward),
        'mean_coverage': float(mean_coverage),
        'episode_rewards': [float(x) for x in callback.episode_rewards],
        'episode_women_reached': [int(x) for x in callback.episode_women_reached],
        'episode_coverage_rates': [float(x) for x in callback.episode_coverage_rates]
    }
    
    os.makedirs("results/ppo", exist_ok=True)
    with open(f"results/ppo/{run_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Mean Reward (last 100 episodes): {mean_reward:.2f}")
    print(f"Mean Coverage (last 100 episodes): {mean_coverage:.2%}")
    
    env.close()
    return model, metrics


if __name__ == "__main__":
    # Only 3 PPO configurations - best algorithm
    ppo_configs = [
        {'learning_rate': 1e-3, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'total_timesteps': 200000, 'run_name': 'ppo_optimized_1'},
        {'learning_rate': 5e-4, 'n_steps': 2048, 'batch_size': 128, 'n_epochs': 15, 'total_timesteps': 200000, 'run_name': 'ppo_optimized_2'},
        {'learning_rate': 3e-4, 'n_steps': 4096, 'batch_size': 64, 'n_epochs': 10, 'total_timesteps': 200000, 'run_name': 'ppo_optimized_3'},
    ]
    
    print("="*80)
    print("TRAINING PPO AGENTS")
    print("="*80)
    for config in ppo_configs:
        train_ppo(**config)