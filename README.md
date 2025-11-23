# ZTPO_Project
AI-powered Zero-Trust Policy Optimization using reinforcement learning: autonomously reduces misconfigurations by 32%, cuts incident response time by 40%, and dynamically adapts access rules in real-time
### Abstract
The Zero-Trust model ("never trust, always verify") is now a cornerstone of modern cybersecurity, but manually managing fine-grained policies at scale is error-prone and static. This research introduces **AI-Driven Zero-Trust Policy Optimization (ZTPO)**—a closed-loop system that continuously learns normal user/device behavior and autonomously tunes access rules using reinforcement learning.

Key results (tested on 30,000+ simulated access events):
- ↓ 32% policy misconfigurations  
- ↓ 37% unauthorized access attempts  
- ↓ 40% average incident response time  
- ↓ 41% false positive rate

### Repository Contents
- `paper/` – Full research paper (PDF + LaTeX source)
- `code/` – Complete implementation (Python + TensorFlow)
  - Data preprocessing & simulation environment
  - Deep Q-Learning (DQN) agent for policy optimization
  - Integration examples with IAM/SIEM APIs
  - Evaluation scripts and notebooks
- `datasets/` – Preprocessed UNSW-NB15 subset + synthetic access logs (anonymized)
- `results/` – Figures, tables, and detailed experiment logs
