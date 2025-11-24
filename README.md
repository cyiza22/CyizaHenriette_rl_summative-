
# **Breast Cancer Awareness Navigator: Reinforcement Learning for Health Outreach Optimization**

## **1. Project Overview**

This project presents a reinforcement learning (RL) approach for optimizing mobile health worker navigation in underserved African communities.
The central objective is to maximize breast cancer awareness outreach by enabling an autonomous agent to efficiently identify and educate women across a spatial environment.

**Mission Statement:**
Improve access to early breast cancer detection knowledge among women in underserved African regions through AI-driven navigation and outreach optimization.


## **2. Performance Summary**

The agent was trained using two categories of RL algorithms—Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO) across multiple hyperparameter configurations.

### **2.1 Top Performing Configuration**

| Metric                        | Value                 |
| ----------------------------- | --------------------- |
| **Best Algorithm**            | PPO (ppo_optimized_1) |
| **Mean Reward**               | **27,637.70**        |
| **Coverage Rate**             | **90.7%**             |
| **Women Reached per Episode** | **2.72 / 3**          |

### **2.2 Algorithm Comparison Overview**

| Algorithm | Configuration       | Mean Reward   | Coverage (%) |
| --------- | ------------------- | ------------- | ------------ |
| **PPO**   | **ppo_optimized_1** | **27,637.70** | **90.7**     |
| PPO       | ppo_optimized_2     | 20,205.95     | 66.7         |
| PPO       | ppo_optimized_3     | 19,521.00     | 65.0         |
| DQN       | dqn_optimized_1     | 42,757.05     | 27.0         |
| DQN       | dqn_optimized_2     | 40,018.60     | 15.0         |
| DQN       | dqn_optimized_3     | 41,897.80     | 5.7          |

**Key Finding:** PPO achieved the best task-aligned performance, strongly outperforming DQN in coverage, despite DQN obtaining higher numerical rewards 
due to its exploitation of movement related reward structures. PPO demonstrated superior policy quality by focusing on reaching and educating women, 
rather than accumulating movement rewards.



## **3. Project Structure**

CyizaHenriette_rl_summative-/


├── environment/                    # Custom Gymnasium environment


├── training/                       # DQN and PPO training scripts


├── models/                         # Saved trained models


├── results/                        # Evaluation metrics and summary tables


├── logs/                           # TensorBoard logs


├── plots/                          # Training and comparison visualizations


├── main.py                         # Inference and visualization runner


├── demo_random_agent.py            # Baseline random agent demonstration


├── analyze_results.py              # Automated analysis and plotting


├── check_results.py                # Quick results inspection


├── requirements.txt                # Dependencies


└── README.md                       # This documentation


## **4. Installation and Setup**

### **4.1 Prerequisites**

* Python ≥ 3.8
* pip
* Git

### **4.2 Environment Setup**


git clone https://github.com/cyiza22/CyizaHenriette_rl_summative.git

cd CyizaHenriette_rl_summative

python -m venv venv

venv\Scripts\activate       # Windows

# or

source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt


## **5. Usage Guide**

### **5.1 Test the Environment (Random Agent)**

python demo_random_agent.py --episodes 3


### **5.2 View Evaluation Results**

python check_results.py

### **5.3 Run the Best Model with Visualization**

python main.py --algorithm ppo --model ppo_optimized_1 --episodes 5


### **5.4 Generate Analysis Plots**

python analyze_results.py


## **6. Environment Specification**

### **6.1 Grid Layout**

A deterministic 5×5 grid with three target locations representing women needing education.

### **6.2 Action Space**

* Move Up
* Move Down
* Move Left
* Move Right
* Educate (available when co-located with a target)

### **6.3 Observation Space (28-D)**

Includes agent position, grid encoding, women reached, and step counter.

### **6.4 Reward Structure**

Reward shaping prioritizes:

* Exploration (+10 base)
* Movement bonuses
* Discovery of women (+1,000)
* Successful education (+5,000)
* Milestones for 1st, 2nd, 3rd woman


## **7. Experimental Findings**

### **7.1 PPO Convergence Pattern**

* Initial exploration: ~5,000 mean reward
* Intermediate competence: ~15,000
* Mature policy: ~27,000

### **7.2 DQN Observations**

* High movement rewards inflated total reward
* Coverage remained low due to insufficient policy quality

### **7.3 Key Insight**

Policy-gradient methods (PPO) exhibit substantially better alignment with the task goal of reaching and educating all targets, 
particularly in sparse-reward and navigation-heavy environments.


## **8. Evaluation Metrics**

* Mean reward over last 100 episodes
* Coverage rate
* Episode success rate
* Steps to completion

**Best Model Performance (ppo_optimized_1)**

* 90.7% coverage
* 27,637 mean reward
* ~91% success rate





