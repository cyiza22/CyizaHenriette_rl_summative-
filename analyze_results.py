"""
Script to analyze training results and generate plots for the report.
Run this after training all models to create visualizations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def load_metrics(algorithm):
    """Load all metrics for a given algorithm."""
    results_dir = Path(f"results/{algorithm}")
    metrics_list = []
    
    for file in results_dir.glob("*_metrics.json"):
        with open(file, 'r') as f:
            metrics = json.load(f)
            metrics['run_name'] = file.stem.replace('_metrics', '')
            metrics_list.append(metrics)
    
    return metrics_list

def plot_cumulative_rewards():
    """Plot cumulative rewards for all algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cumulative Rewards Across All Algorithms', fontsize=16, fontweight='bold')
    
    algorithms = ['dqn', 'reinforce', 'a2c', 'ppo']
    titles = ['DQN', 'REINFORCE', 'A2C', 'PPO']
    
    for idx, (algo, title) in enumerate(zip(algorithms, titles)):
        ax = axes[idx // 2, idx % 2]
        
        metrics_list = load_metrics(algo)
        
        # Find best run
        best_run = max(metrics_list, key=lambda x: x.get('mean_reward', 0))
        
        # Plot rewards
        rewards = best_run.get('episode_rewards', [])
        if rewards:
            # Calculate rolling mean
            window = min(50, len(rewards) // 10)
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            
            ax.plot(rewards, alpha=0.3, color='lightblue', label='Episode Reward')
            ax.plot(range(window-1, len(rewards)), rolling_mean, 
                   color='darkblue', linewidth=2, label=f'Rolling Mean (window={window})')
            
            ax.set_title(f'{title} - Best Configuration: {best_run["run_name"]}', 
                        fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Cumulative Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add mean reward text
            mean_reward = best_run.get('mean_reward', 0)
            ax.text(0.02, 0.98, f'Mean Reward: {mean_reward:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('plots/cumulative_rewards_all.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/cumulative_rewards_all.png")
    plt.close()

def plot_algorithm_comparison():
    """Compare best runs from each algorithm."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    algorithms = ['dqn', 'reinforce', 'a2c', 'ppo']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Collect best runs
    best_runs = {}
    for algo in algorithms:
        metrics_list = load_metrics(algo)
        best_run = max(metrics_list, key=lambda x: x.get('mean_reward', 0))
        best_runs[algo] = best_run
    
    # Plot 1: Mean Reward Comparison
    ax1 = axes[0]
    algos_upper = [a.upper() for a in algorithms]
    mean_rewards = [best_runs[a].get('mean_reward', 0) for a in algorithms]
    
    bars = ax1.bar(algos_upper, mean_rewards, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('Mean Reward Comparison (Best Configuration per Algorithm)', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Reward (Last 100 Episodes)', fontsize=12)
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, reward in zip(bars, mean_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Coverage Rate Comparison
    ax2 = axes[1]
    coverage_rates = [best_runs[a].get('mean_coverage', 0) * 100 for a in algorithms]
    
    bars = ax2.bar(algos_upper, coverage_rates, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_title('Mean Coverage Rate Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean Coverage Rate (%)', fontsize=12)
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, coverage in zip(bars, coverage_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{coverage:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/algorithm_comparison.png")
    plt.close()

def plot_convergence():
    """Plot episodes to convergence for each algorithm."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    algorithms = ['dqn', 'reinforce', 'a2c', 'ppo']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for algo, color in zip(algorithms, colors):
        metrics_list = load_metrics(algo)
        best_run = max(metrics_list, key=lambda x: x.get('mean_reward', 0))
        
        rewards = best_run.get('episode_rewards', [])
        if rewards:
            # Calculate rolling mean
            window = 100
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            
            ax.plot(range(window-1, len(rewards)), rolling_mean, 
                   label=f'{algo.upper()} ({best_run["run_name"]})',
                   linewidth=2, color=color)
    
    ax.set_title('Convergence Comparison (Rolling Mean Over 100 Episodes)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Mean Reward (100-episode window)', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/convergence_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/convergence_comparison.png")
    plt.close()

def plot_hyperparameter_sensitivity(algorithm='ppo'):
    """Plot hyperparameter sensitivity for a specific algorithm."""
    metrics_list = load_metrics(algorithm)
    
    if algorithm == 'ppo':
        # Extract learning rates and rewards
        lr_configs = [(m['learning_rate'], m['mean_reward'], m['run_name']) 
                     for m in metrics_list if 'lr' in m['run_name'].lower()]
        
        if lr_configs:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            lrs = [c[0] for c in lr_configs]
            rewards = [c[1] for c in lr_configs]
            names = [c[2] for c in lr_configs]
            
            ax.scatter(lrs, rewards, s=100, c='darkblue', edgecolor='black', linewidth=1.5)
            
            for lr, reward, name in zip(lrs, rewards, names):
                ax.annotate(name, (lr, reward), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            ax.set_title(f'PPO Learning Rate Sensitivity', fontsize=14, fontweight='bold')
            ax.set_xlabel('Learning Rate', fontsize=12)
            ax.set_ylabel('Mean Reward', fontsize=12)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('plots/ppo_lr_sensitivity.png', dpi=300, bbox_inches='tight')
            print("Saved: plots/ppo_lr_sensitivity.png")
            plt.close()

def generate_summary_table():
    """Generate a summary table of all experiments."""
    algorithms = ['dqn', 'reinforce', 'a2c', 'ppo']
    
    summary = []
    for algo in algorithms:
        metrics_list = load_metrics(algo)
        
        for metrics in metrics_list:
            summary.append({
                'Algorithm': algo.upper(),
                'Configuration': metrics['run_name'],
                'Mean Reward': metrics.get('mean_reward', 0),
                'Mean Coverage': metrics.get('mean_coverage', 0) * 100
            })
    
    # Sort by mean reward
    summary.sort(key=lambda x: x['Mean Reward'], reverse=True)
    
    # Save to file
    with open('results/summary_table.txt', 'w') as f:
        f.write(f"{'Algorithm':<12} {'Configuration':<30} {'Mean Reward':<15} {'Coverage %':<12}\n")
        f.write("="*70 + "\n")
        for item in summary:
            f.write(f"{item['Algorithm']:<12} {item['Configuration']:<30} "
                   f"{item['Mean Reward']:<15.2f} {item['Mean Coverage']:<12.1f}\n")
    
    print("Saved: results/summary_table.txt")
    
    # Also print top 5
    print("\nTop 5 Configurations:")
    print(f"{'Algorithm':<12} {'Configuration':<30} {'Mean Reward':<15} {'Coverage %'}")
    print("="*70)
    for item in summary[:5]:
        print(f"{item['Algorithm']:<12} {item['Configuration']:<30} "
              f"{item['Mean Reward']:<15.2f} {item['Mean Coverage']:.1f}%")

def main():
    """Generate all plots and analysis."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    print("Generating plots and analysis...")
    print("="*80)
    
    try:
        plot_cumulative_rewards()
        plot_algorithm_comparison()
        plot_convergence()
        plot_hyperparameter_sensitivity('ppo')
        generate_summary_table()
        
        print("\n" + "="*80)
        print("All plots and analysis generated successfully!")
        print("Check the 'plots/' directory for visualizations.")
        print("="*80)
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()