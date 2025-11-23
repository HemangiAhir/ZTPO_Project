"""
Main training script for ZTPO model
"""
from ztpo_training import train_ztpo_model

if __name__ == "__main__":
    # Train the model
    agent, integrator, scores, losses = train_ztpo_model(
        unsw_path='/home/kali/ZTPO_Project/datasets/UNSW_NB15_combined.csv',
        cicids_path='/home/kali/ZTPO_Project/datasets/CICIDS2017_combined.csv',
        cert_path='/home/kali/ZTPO_Project/datasets/merged_dataset.csv',
        episodes=20
    )
    
    print("\nTraining completed successfully!")
    print(f"Final episode score: {scores[-1]:.2f}")
    print(f"Average loss: {sum(losses)/len(losses):.4f}")