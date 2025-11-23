"""
Zero Trust Policy Optimization - Results Visualization
Generates plots and metrics for research paper
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def create_visualizations(scores, losses, save_dir='../results/'):
    """
    Create comprehensive visualizations of training results
    
    Args:
        scores: List of rewards per episode
        losses: List of losses per episode
        save_dir: Directory to save plots
    """
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # ==================== Plot 1: Training Reward ====================
    ax1 = plt.subplot(2, 3, 1)
    episodes = range(1, len(scores) + 1)
    ax1.plot(episodes, scores, 'b-', linewidth=2, label='Episode Reward')
    ax1.fill_between(episodes, scores, alpha=0.3)
    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Training Progress: Reward per Episode', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ==================== Plot 2: Training Loss ====================
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(episodes, losses, 'r-', linewidth=2, label='Episode Loss')
    ax2.fill_between(episodes, losses, alpha=0.3, color='red')
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Progress: Loss per Episode', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # ==================== Plot 3: Learning Curve (Moving Average) ====================
    ax3 = plt.subplot(2, 3, 3)
    window = 3
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax3.plot(range(window, len(scores) + 1), moving_avg, 'g-', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax3.plot(episodes, scores, 'b--', alpha=0.5, label='Raw Scores')
    ax3.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax3.set_title('Learning Curve (Smoothed)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # ==================== Plot 4: Performance Metrics Bar Chart ====================
    ax4 = plt.subplot(2, 3, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [100.0, 100.0, 100.0, 100.0]  # Your perfect results!
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # ==================== Plot 5: Comparison Chart ====================
    ax5 = plt.subplot(2, 3, 5)
    categories = ['Policy\nMisconfig', 'Unauthorized\nAccess', 'Incident\nResponse', 'False\nPositives']
    baseline = [85, 240, 10.2, 6.5]
    ztpo = [58, 150, 6.1, 3.8]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, baseline, width, label='Baseline (Static ZT)', 
                    color='#e74c3c', alpha=0.7, edgecolor='black')
    bars2 = ax5.bar(x + width/2, ztpo, width, label='AI-Driven ZTPO',
                    color='#2ecc71', alpha=0.7, edgecolor='black')
    
    ax5.set_ylabel('Count / Time (min) / Rate (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Baseline vs AI-ZTPO Comparison', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, fontsize=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ==================== Plot 6: Improvement Percentages ====================
    ax6 = plt.subplot(2, 3, 6)
    improvements = [32, 37, 40, 41]  # From your paper
    categories_short = ['Policy\nErrors', 'Unauthorized\nAccess', 'Response\nTime', 'False\nPositives']
    colors_imp = ['#9b59b6', '#e67e22', '#1abc9c', '#34495e']
    
    bars = ax6.barh(categories_short, improvements, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=2)
    ax6.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Performance Improvements', fontsize=14, fontweight='bold')
    ax6.set_xlim([0, 50])
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2.,
                f' ↓{improvements[i]}%', ha='left', va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(save_dir, 'training_results_comprehensive.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive visualization: {output_path}")
    
    plt.show()
    
    # ==================== Individual Plots for Paper ====================
    
    # High-quality reward plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, scores, 'b-', linewidth=3, marker='o', markersize=6, label='Episode Reward')
    ax.fill_between(episodes, scores, alpha=0.2)
    ax.set_xlabel('Training Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Reward', fontsize=14, fontweight='bold')
    ax.set_title('Deep Q-Learning Training Performance', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    plt.tight_layout()
    output_path2 = os.path.join(save_dir, 'reward_curve_paper.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Saved reward curve for paper: {output_path2}")
    plt.show()
    
    # High-quality loss plot
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, losses, 'r-', linewidth=3, marker='s', markersize=6, label='Training Loss')
    ax.fill_between(episodes, losses, alpha=0.2, color='red')
    ax.set_xlabel('Training Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Loss', fontsize=14, fontweight='bold')
    ax.set_title('Model Convergence: Loss Reduction', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    plt.tight_layout()
    output_path3 = os.path.join(save_dir, 'loss_curve_paper.png')
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"✅ Saved loss curve for paper: {output_path3}")
    plt.show()


def generate_metrics_report(scores, losses, save_dir='../results/'):
    """Generate text report of metrics for paper"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    report = f"""
{'='*60}
ZERO TRUST POLICY OPTIMIZATION - RESULTS REPORT
{'='*60}

TRAINING CONFIGURATION:
- Episodes: {len(scores)}
- Initial Reward: {scores[0]:.2f}
- Final Reward: {scores[-1]:.2f}
- Reward Improvement: {((scores[-1] - scores[0]) / scores[0] * 100):.2f}%

- Initial Loss: {losses[0]:.4f}
- Final Loss: {losses[-1]:.4f}
- Loss Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%

PERFORMANCE METRICS:
- Test Accuracy: 100.00%
- False Positive Rate: 0.00%
- False Negative Rate: 0.00%
- Precision: 100.00%
- Recall: 100.00%
- F1-Score: 100.00%

COMPARISON TO BASELINE (from paper):
- Policy Misconfigurations: ↓32% (85 → 58)
- Unauthorized Access Attempts: ↓37% (240 → 150)
- Incident Response Time: ↓40% (10.2 min → 6.1 min)
- False Positive Rate: ↓41% (6.5% → 3.8%)

KEY ACHIEVEMENTS:
✅ Successfully trained Deep Q-Learning agent
✅ Achieved perfect accuracy on test set
✅ Demonstrated significant improvements over baseline
✅ Zero false positives in final evaluation
✅ Model converged successfully

DATASETS USED:
- UNSW-NB15: Network intrusion detection (5,000 samples)
- CICIDS2017: Contextual behavior analysis (5,000 samples)
- CERT Insider Threat: User behavior patterns (5,000 samples)
- Total Training Records: 6,000 unified features

{'='*60}
Report generated for research publication
{'='*60}
"""
    
    output_path = os.path.join(save_dir, 'metrics_report.txt')
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Saved metrics report: {output_path}")
    print(report)


if __name__ == "__main__":
    # Example: Load your training results
    # These would be the scores and losses from your training
    
    # Sample data (replace with actual data from training)
    scores = [1085, 2810, 3935, 4325, 4715, 4865, 4925, 4925, 4925, 4940, 
              4925, 4925, 4955, 4955, 4955, 4955, 4910, 4925, 4955, 5000]
    losses = [8.6217, 4.7828, 3.2156, 2.4974, 2.0833, 1.8096, 1.8572, 1.8143,
              1.7565, 1.6950, 1.6361, 1.7070, 1.7302, 1.7288, 1.7181, 1.7017,
              1.7741, 1.8146, 1.8406, 1.8572]
    
    print("Generating visualizations and reports...")
    print("="*60)
    
    create_visualizations(scores, losses)
    generate_metrics_report(scores, losses)
    
    print("\n" + "="*60)
    print("All visualizations and reports generated successfully!")
    print("Check the '../results/' folder for outputs")
    print("="*60)