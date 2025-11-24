import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import json

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
        # Check if episode ended
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            # Get the actual episode reward from the monitor wrapper
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
            self.episode_women_reached.append(info.get('women_reached', 0))
            self.episode_coverage_rates.append(info.get('coverage_rate', 0))
        return True


def train_dqn(
    learning_rate=5e-4,
    gamma=0.99,
    buffer_size=50000,
    batch_size=128,
    exploration_fraction=0.5,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    target_update_interval=500,
    train_freq=4,
    gradient_steps=1,
    total_timesteps=200000,
    run_name="dqn_run"
):
    """
    Train DQN agent with specified hyperparameters.
    """
    
    # Create environment
    env = BreastCancerAwarenessEnv(render_mode=None)
    env = Monitor(env)
    
    # Create DQN model with optimized settings
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        learning_starts=1000,  # Start learning after 1000 steps
        policy_kwargs=dict(net_arch=[256, 256]),  # Bigger network
        verbose=1,
        tensorboard_log=f"./logs/dqn/{run_name}"
    )
    
    # Create callback
    callback = MetricsCallback()
    
    print(f"\nTraining DQN with configuration:")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Buffer Size: {buffer_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Exploration Fraction: {exploration_fraction}")
    print(f"Initial Epsilon: {exploration_initial_eps}")
    print(f"Final Epsilon: {exploration_final_eps}")
    print(f"Target Update Interval: {target_update_interval}")
    print(f"Total Timesteps: {total_timesteps}\n")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
        progress_bar=True
    )
    
    # Save model
    os.makedirs("models/dqn", exist_ok=True)
    model.save(f"models/dqn/{run_name}")
    
    # Calculate metrics
    mean_reward = np.mean(callback.episode_rewards[-100:]) if len(callback.episode_rewards) > 0 else 0
    mean_coverage = np.mean(callback.episode_coverage_rates[-100:]) if len(callback.episode_coverage_rates) > 0 else 0
    
    # Save metrics - Convert to Python native types
    metrics = {
        'learning_rate': learning_rate,
        'gamma': gamma,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'exploration_fraction': exploration_fraction,
        'exploration_initial_eps': exploration_initial_eps,
        'exploration_final_eps': exploration_final_eps,
        'target_update_interval': target_update_interval,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'total_timesteps': total_timesteps,
        'mean_reward': float(mean_reward),
        'mean_coverage': float(mean_coverage),
        'episode_rewards': [float(x) for x in callback.episode_rewards],
        'episode_women_reached': [int(x) for x in callback.episode_women_reached],
        'episode_coverage_rates': [float(x) for x in callback.episode_coverage_rates]
    }
    
    os.makedirs("results/dqn", exist_ok=True)
    with open(f"results/dqn/{run_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Mean Reward (last 100 episodes): {mean_reward:.2f}")
    print(f"Mean Coverage (last 100 episodes): {mean_coverage:.2%}")
    
    env.close()
    
    return model, metrics


if __name__ == "__main__":
    # Only 3 GOOD configurations - quality over quantity
    configs = [
        # High learning rate, more exploration
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'buffer_size': 50000,
            'batch_size': 128,
            'exploration_fraction': 0.6,
            'exploration_final_eps': 0.1,
            'target_update_interval': 500,
            'total_timesteps': 200000,
            'run_name': 'dqn_optimized_1'
        },
        # Balanced
        {
            'learning_rate': 5e-4,
            'gamma': 0.98,
            'buffer_size': 50000,
            'batch_size': 128,
            'exploration_fraction': 0.5,
            'exploration_final_eps': 0.05,
            'target_update_interval': 1000,
            'total_timesteps': 200000,
            'run_name': 'dqn_optimized_2'
        },
        # More stable
        {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'buffer_size': 100000,
            'batch_size': 256,
            'exploration_fraction': 0.4,
            'exploration_final_eps': 0.02,
            'target_update_interval': 1000,
            'total_timesteps': 200000,
            'run_name': 'dqn_optimized_3'
        }
    ]
    
    # Train all configurations
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Training: {config['run_name']}")
        print(f"{'='*80}")
        train_dqn(**config)