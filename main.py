"""
Main entry point for running the best performing trained agent.
This script loads the best model and runs a simulation with visualization.
"""

import os
import sys
import json
import time
import argparse
from stable_baselines3 import DQN, PPO, A2C

from environment.custom_env import BreastCancerAwarenessEnv


def load_best_model(algorithm='ppo', model_name=None):
    """
    Load the best performing model based on saved metrics.
    
    Args:
        algorithm: Algorithm type ('dqn', 'ppo', 'a2c')
        model_name: Specific model name to load (optional)
    
    Returns:
        Loaded model and its metrics
    """
    
    results_dir = f"results/{algorithm}"
    
    if model_name:
        # Load specific model
        metrics_path = f"{results_dir}/{model_name}_metrics.json"
        model_path = f"models/{algorithm}/{model_name}"
    else:
        # Find best model based on mean coverage
        best_coverage = -float('inf')
        best_model_name = None
        
        for filename in os.listdir(results_dir):
            if filename.endswith('_metrics.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    metrics = json.load(f)
                    coverage = metrics.get('mean_coverage', 0)
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_model_name = filename.replace('_metrics.json', '')
        
        if best_model_name is None:
            raise ValueError(f"No trained models found for {algorithm}")
        
        print(f"Loading best {algorithm.upper()} model: {best_model_name}")
        print(f"Mean Coverage: {best_coverage*100:.1f}%")
        
        metrics_path = f"{results_dir}/{best_model_name}_metrics.json"
        model_path = f"models/{algorithm}/{best_model_name}"
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load model
    if algorithm == 'dqn':
        model = DQN.load(model_path)
    elif algorithm == 'ppo':
        model = PPO.load(model_path)
    elif algorithm == 'a2c':
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return model, metrics


def run_simulation(algorithm='ppo', model_name=None, num_episodes=5, render=True):
    """
    Run simulation with trained agent.
    
    Args:
        algorithm: Algorithm type
        model_name: Specific model name
        num_episodes: Number of episodes to run
        render: Whether to render visualization
    """
    
    # Load model
    model, metrics = load_best_model(algorithm, model_name)
    
    # Create environment
    render_mode = 'human' if render else None
    env = BreastCancerAwarenessEnv(render_mode=render_mode)
    
    print(f"\n{'='*80}")
    print(f"RUNNING SIMULATION - {algorithm.upper()}")
    print(f"{'='*80}")
    print(f"\nModel Hyperparameters:")
    for key, value in metrics.items():
        if key not in ['episode_rewards', 'episode_women_reached', 'episode_coverage_rates']:
            print(f"  {key}: {value}")
    
    print(f"\n{'='*80}")
    print("Starting Episodes...")
    print(f"{'='*80}\n")
    
    episode_results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        while not done:
            # Select action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            if render:
                env.render()
                time.sleep(0.05)  # Slow down for visibility
        
        # Print episode summary
        print(f"Steps: {step_count}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Women Reached: {info['women_reached']}/{info['total_women']}")
        print(f"Coverage Rate: {info['coverage_rate']:.2%}")
        
        episode_results.append({
            'episode': episode + 1,
            'steps': step_count,
            'reward': episode_reward,
            'women_reached': info['women_reached'],
            'coverage_rate': info['coverage_rate']
        })
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("SIMULATION SUMMARY")
    print(f"{'='*80}")
    avg_reward = sum(r['reward'] for r in episode_results) / num_episodes
    avg_coverage = sum(r['coverage_rate'] for r in episode_results) / num_episodes
    avg_steps = sum(r['steps'] for r in episode_results) / num_episodes
    total_women = sum(r['women_reached'] for r in episode_results)
    
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Coverage: {avg_coverage:.2%}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Total Women Reached: {total_women}/{num_episodes * 3}")
    print(f"{'='*80}\n")
    
    env.close()
    
    return episode_results


def main():
    parser = argparse.ArgumentParser(description='Run trained RL agent for Breast Cancer Awareness')
    parser.add_argument('--algorithm', type=str, default='ppo', 
                       choices=['dqn', 'ppo', 'a2c'],
                       help='Algorithm to use')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model name to load')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    run_simulation(
        algorithm=args.algorithm,
        model_name=args.model,
        num_episodes=args.episodes,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()