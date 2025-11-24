import json
from pathlib import Path

print('='*80)
print('PPO RESULTS SUMMARY')
print('='*80)

results = []
for file in Path('results/ppo').glob('*_metrics.json'):
    with open(file) as f:
        data = json.load(f)
        results.append({
            'name': file.stem.replace('_metrics', ''),
            'reward': data['mean_reward'],
            'coverage': data['mean_coverage'] * 100
        })

# Sort by coverage
results.sort(key=lambda x: x['coverage'], reverse=True)

print(f"{'Configuration':<30} | {'Mean Reward':>12} | {'Coverage':>10}")
print('-'*80)
for r in results:
    print(f"{r['name']:<30} | {r['reward']:>12.2f} | {r['coverage']:>9.1f}%")

print('='*80)
print(f"\nBEST MODEL: {results[0]['name']}")
print(f"Coverage: {results[0]['coverage']:.1f}%")
print(f"Reward: {results[0]['reward']:.2f}")