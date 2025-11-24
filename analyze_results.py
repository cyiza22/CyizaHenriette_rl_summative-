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
    
    if not results_dir.exists():
        return []
    
    metrics_list = []
    
    for file in results_dir.glob("*_metrics.json"):
        try:
            with open(file, 'r') as f:
                metrics = json.load(f)
                metrics['run_name'] = file.stem.replace('_metrics', '')
                metrics_list.append(metrics)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return metrics_list

def plot_cumulative_rewards():
    """Plot cumulative rewards for available algorithms."""
    
    # Check which algorithms have results
    available_algos = []
    algo_names = {'dqn': 'DQN', 'ppo': 'PPO', 'a2c': 'A2C', 'reinforce': 'REINFORCE'}
    
    for algo in ['dqn', 'ppo', 'a2c', 'reinforce']:
        if Path(f"results/{algo}").exists() and list(Path(f"results/{algo}").glob("*_metrics.json")):
            available_algos.append(algo)
    
    if not available_algos:
        print("No results found!")
        return
    
    num_algos = len(available_algos)
    fig, axes = plt.subplots(1, num_algos, figsize=(8*num_algos, 6))
    
    if num_algos == 1:
        axes = [axes]
    
    fig.suptitle('Cumulative Rewards Across Algorithms', fontsize=16, fontweight='bold')
    
    for idx, algo in enumerate(available_algos):
        ax = axes[idx]
        
        metrics_list = load_metrics(algo)
        
        if not metrics_list:
            continue
        
        # Find best run
        best_run = max(metrics_list, key=lambda x: x.get('mean_coverage', 0))
        
        # Plot rewards
        rewards = best_run.get('episode_rewards', [])
        if rewards:
            # Calculate rolling mean
            window = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
            if window > 1:
                rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(rewards, alpha=0.3, color='lightblue', label='Episode Reward')
                ax.plot(range(window-1, len(rewards)), rolling_mean, 
                       color='darkblue', linewidth=2, label=f'Rolling Mean (window={window})')
            else:
                ax.plot(rewards, color='darkblue', linewidth=2, label='Episode Reward')
            
            ax.set_title(f'{algo_names[algo]} - Best: {best_run["run_name"]}', 
                        fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Cumulative Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics text
            mean_reward = best_run.get('mean_reward', 0)
            mean_coverage = best_run.get('mean_coverage', 0)
            ax.text(0.02, 0.98, f'Mean Reward: {mean_reward:.2f}\nCoverage: {mean_coverage:.1%}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('plots/cumulative_rewards_all.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/cumulative_rewards_all.png")
    plt.close()

def plot_algorithm_comparison():
    """Compare best runs from each available algorithm."""
    
    # Check which algorithms have results
    available_algos = []
    for algo in ['dqn', 'ppo', 'a2c', 'reinforce']:
        if Path(f"results/{algo}").exists() and list(Path(f"results/{algo}").glob("*_metrics.json")):
            available_algos.append(algo)
    
    if not available_algos:
        print("No results found!")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'dqn': '#FF6B6B', 'reinforce': '#4ECDC4', 'a2c': '#45B7D1', 'ppo': '#96CEB4'}
    
    # Collect best runs
    best_runs = {}
    for algo in available_algos:
        metrics_list = load_metrics(algo)
        if metrics_list:
            best_run = max(metrics_list, key=lambda x: x.get('mean_coverage', 0))
            best_runs[algo] = best_run
    
    if not best_runs:
        print("No valid results found!")
        return
    
    # Plot 1: Mean Reward Comparison
    ax1 = axes[0]
    algos_upper = [a.upper() for a in best_runs.keys()]
    mean_rewards = [best_runs[a].get('mean_reward', 0) for a in best_runs.keys()]
    bar_colors = [colors.get(a, '#999999') for a in best_runs.keys()]
    
    bars = ax1.bar(algos_upper, mean_rewards, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('Mean Reward Comparison (Best Configuration per Algorithm)', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Reward (Last 100 Episodes)', fontsize=12)
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, reward in zip(bars, mean_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.0f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Coverage Rate Comparison
    ax2 = axes[1]
    coverage_rates = [best_runs[a].get('mean_coverage', 0) * 100 for a in best_runs.keys()]
    
    bars = ax2.bar(algos_upper, coverage_rates, color=bar_colors, edgecolor='black', linewidth=1.5)
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
    
    available_algos = []
    for algo in ['dqn', 'ppo', 'a2c', 'reinforce']:
        if Path(f"results/{algo}").exists() and list(Path(f"results/{algo}").glob("*_metrics.json")):
            available_algos.append(algo)
    
    if not available_algos:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'dqn': '#FF6B6B', 'reinforce': '#4ECDC4', 'a2c': '#45B7D1', 'ppo': '#96CEB4'}
    
    for algo in available_algos:
        metrics_list = load_metrics(algo)
        if not metrics_list:
            continue
            
        best_run = max(metrics_list, key=lambda x: x.get('mean_coverage', 0))
        
        rewards = best_run.get('episode_rewards', [])
        if len(rewards) > 100:
            # Calculate rolling mean
            window = 100
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            
            ax.plot(range(window-1, len(rewards)), rolling_mean, 
                   label=f'{algo.upper()} ({best_run["run_name"]})',
                   linewidth=2, color=colors.get(algo, '#999999'))
    
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

def generate_summary_table():
    """Generate a summary table of all experiments."""
    
    available_algos = []
    for algo in ['dqn', 'ppo', 'a2c', 'reinforce']:
        if Path(f"results/{algo}").exists() and list(Path(f"results/{algo}").glob("*_metrics.json")):
            available_algos.append(algo)
    
    summary = []
    for algo in available_algos:
        metrics_list = load_metrics(algo)
        
        for metrics in metrics_list:
            summary.append({
                'Algorithm': algo.upper(),
                'Configuration': metrics['run_name'],
                'Mean Reward': metrics.get('mean_reward', 0),
                'Mean Coverage': metrics.get('mean_coverage', 0) * 100
            })
    
    # Sort by coverage
    summary.sort(key=lambda x: x['Mean Coverage'], reverse=True)
    
    # Save to file
    with open('results/summary_table.txt', 'w') as f:
        f.write(f"{'Algorithm':<12} {'Configuration':<30} {'Mean Reward':<15} {'Coverage %':<12}\n")
        f.write("="*70 + "\n")
        for item in summary:
            f.write(f"{item['Algorithm']:<12} {item['Configuration']:<30} "
                   f"{item['Mean Reward']:<15.2f} {item['Mean Coverage']:<12.1f}\n")
    
    print("Saved: results/summary_table.txt")
    
    # Also print top 10
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS:")
    print("="*80)
    print(f"{'Algorithm':<12} {'Configuration':<30} {'Mean Reward':<15} {'Coverage %'}")
    print("-"*80)
    for item in summary[:10]:
        print(f"{item['Algorithm']:<12} {item['Configuration']:<30} "
              f"{item['Mean Reward']:<15.2f} {item['Mean Coverage']:.1f}%")
    print("="*80)

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