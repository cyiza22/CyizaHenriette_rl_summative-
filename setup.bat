@echo off
echo Setting up RL Summative Project...

:: Create directories
mkdir environment training models\dqn models\ppo models\a2c models\reinforce
mkdir results\dqn results\ppo results\a2c results\reinforce
mkdir logs\dqn logs\ppo logs\a2c logs\reinforce
mkdir plots

:: Create empty __init__.py files
type nul > environment\__init__.py
type nul > training\__init__.py

:: Create virtual environment
python -m venv venv
call venv\Scripts\activate

:: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete!
echo Next steps:
echo 1. Copy Python files from artifacts to their respective locations
echo 2. Run: python demo_random_agent.py --episodes 1
echo 3. Start training: python training/dqn_training.py
```

---

## File Sizes Reference

After training, expect these approximate sizes:
```
yourname_rl_summative/
├── environment/
│   └── custom_env.py              (~15 KB)
├── training/
│   ├── dqn_training.py            (~8 KB)
│   └── pg_training.py             (~12 KB)
├── models/                        (~200-400 MB total)
│   ├── dqn/                       (~50 MB - 10 models)
│   ├── ppo/                       (~100 MB - 10 models)
│   ├── a2c/                       (~50 MB - 10 models)
│   └── reinforce/                 (~10 MB - 10 models)
├── results/                       (~5 MB total)
│   ├── dqn/                       (~1 MB - 10 JSON files)
│   ├── ppo/                       (~2 MB - 10 JSON files)
│   ├── a2c/                       (~1 MB - 10 JSON files)
│   └── reinforce/                 (~1 MB - 10 JSON files)
├── logs/                          (~100-200 MB - TensorBoard logs)
├── plots/                         (~2-5 MB - PNG images)
├── main.py                        (~5 KB)
├── demo_random_agent.py           (~5 KB)
├── analyze_results.py             (~7 KB)
├── requirements.txt               (~1 KB)
├── README.md                      (~5 KB)
├── SETUP_GUIDE.md                 (~15 KB)
├── QUICK_REFERENCE.md             (~10 KB)
└── .gitignore                     (~1 KB)