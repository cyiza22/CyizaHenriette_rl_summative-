"""
Demonstration script showing the agent taking random actions in the environment.
This is used to create the static visualization file required by the assignment.
"""

import sys
import time
import numpy as np
from environment.custom_env import BreastCancerAwarenessEnv


def demo_random_agent(num_episodes=3, render=True, delay=0.3):
    """
    Run episodes with random actions to demonstrate the environment visualization.
    
    Args:
        num_episodes: Number of episodes to run
        render: Whether to render the visualization
        delay: Delay between actions (seconds) for better visibility
    """
    
    # Create environment with rendering
    render_mode = 'human' if render else None
    env = BreastCancerAwarenessEnv(render_mode=render_mode)
    
    print("="*80)
    print("BREAST CANCER AWARENESS NAVIGATOR - RANDOM AGENT DEMONSTRATION")
    print("="*80)
    print("\nEnvironment Details:")
    print(f"  Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"  Action Space: {env.action_space.n} discrete actions")
    print(f"    0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: EDUCATE")
    print(f"  Observation Space: {env.observation_space.shape[0]} dimensions")
    print(f"  Max Steps per Episode: {env.max_steps}")
    print("\nReward Structure:")
    print(f"  +1000: Educate a woman")
    print(f"  +50: Move closer to woman")
    print(f"  +500/+800/+1500: Milestone bonuses")
    print("\nLegend:")
    print(f"  🔵 (Blue): Mobile Health Worker (Agent)")
    print(f"  👩 (Pink): Woman needing awareness education")
    print(f"  ✓ (Green): Woman already educated")
    print("="*80)
    print("\nStarting demonstration with RANDOM actions...")
    print("NOTE: Random agent will perform poorly - this demonstrates the visualization.\n")
    
    for episode in range(num_episodes):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*80}")
        
        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'EDUCATE']
        
        while not done and step < 50:  # Limit steps for demo
            # Take random action
            action = env.action_space.sample()
            
            # Print action
            print(f"Step {step + 1}: Action = {action_names[action]}", end="")
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Print result
            print(f" | Reward = {reward:+.1f} | Women Reached = {info['women_reached']}/{info['total_women']}")
            
            # Render
            if render:
                env.render()
                time.sleep(delay)
            
            step += 1
        
        # Episode summary
        print(f"\n{'-'*80}")
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Women Reached: {info['women_reached']}/{info['total_women']}")
        print(f"  Coverage Rate: {info['coverage_rate']:.1%}")
        print(f"{'-'*80}")
        
        if episode < num_episodes - 1:
            print("\nPress Enter to start next episode...")
            input()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED")
    print("="*80)
    print("\nKey Observations from Random Agent:")
    print("  • Random movements lead to inefficient navigation")
    print("  • EDUCATE action used at wrong locations, wasting opportunities")
    print("  • Low coverage rate due to lack of strategy")
    print("\nA trained RL agent will learn to:")
    print("  ✓ Navigate efficiently to reach all women")
    print("  ✓ Use EDUCATE action only when positioned correctly")
    print("  ✓ Achieve high coverage rates (>90%)")
    print("="*80)
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demonstrate random agent in environment')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--delay', type=float, default=0.3,
                       help='Delay between actions (seconds)')
    
    args = parser.parse_args()
    
    demo_random_agent(
        num_episodes=args.episodes,
        render=not args.no_render,
        delay=args.delay
    )