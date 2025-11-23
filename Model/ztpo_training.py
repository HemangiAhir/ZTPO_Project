"""
AI-Driven Zero Trust Policy Optimization (ZTPO) Training Pipeline
Integrates UNSW-NB15, CICIDS2017, and CERT Insider Threat datasets
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import deque
import random
import pickle
import os

# ==================== DATA PREPROCESSING ====================

class DatasetIntegrator:
    """Integrates and preprocesses multiple datasets for ZTPO"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_unsw_nb15(self, filepath):
        """Load and preprocess UNSW-NB15 dataset"""
        print("Loading UNSW-NB15 dataset...")
        # Load only first 5000 rows to save memory and speed up
        df = pd.read_csv(filepath, low_memory=False, nrows=5000)
        print(f"  Loaded {len(df)} rows from UNSW-NB15")
        
        # Key features for network threat detection
        network_features = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 
            'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
            'sloss', 'dloss', 'service', 'sload', 'dload',
            'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
            'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
            'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt',
            'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
            'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
            'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
        ]
        
        # Use available columns
        available_cols = [col for col in network_features if col in df.columns]
        df = df[available_cols].copy()
        
        # Create threat score (0 = benign, 1 = malicious)
        if 'label' in df.columns:
            df['threat_score'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
        else:
            df['threat_score'] = 0
        
        # Feature engineering - handle missing columns
        if 'sbytes' in df.columns and 'dbytes' in df.columns:
            df['traffic_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
        else:
            df['traffic_ratio'] = 0
        
        if 'spkts' in df.columns and 'dur' in df.columns:
            df['packet_rate'] = df['spkts'] / (df['dur'] + 0.001)
        else:
            df['packet_rate'] = 0
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def load_cicids2017(self, filepath):
        """Load and preprocess CICIDS2017 dataset"""
        print("Loading CICIDS2017 dataset...")
        # Load only first 5000 rows to save memory and speed up
        df = pd.read_csv(filepath, low_memory=False, nrows=5000)
        print(f"  Loaded {len(df)} rows from CICIDS2017")
        
        # Standardize column names (remove spaces)
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Key contextual features
        context_features = [
            'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
            'Flow_Bytes/s', 'Flow_Packets/s', 'Flow_IAT_Mean',
            'Fwd_IAT_Total', 'Bwd_IAT_Total', 'Fwd_Header_Length',
            'Bwd_Header_Length', 'Fwd_Packets/s', 'Bwd_Packets/s',
            'Packet_Length_Mean', 'Packet_Length_Std', 'Packet_Length_Variance',
            'Down/Up_Ratio', 'Average_Packet_Size', 'Subflow_Fwd_Packets',
            'Subflow_Bwd_Packets', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward', 'Active_Mean', 'Idle_Mean', 'Label'
        ]
        
        available_cols = [col for col in context_features if col in df.columns]
        df = df[available_cols].copy()
        
        # Create behavior context score
        if 'Label' in df.columns:
            df['behavior_anomaly'] = df['Label'].apply(
                lambda x: 0 if str(x).upper() == 'BENIGN' else 1
            )
        else:
            df['behavior_anomaly'] = 0
        
        # Fill NaN and infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    def load_cert_insider(self, filepath):
        """Load and preprocess CERT Insider Threat dataset"""
        print("Loading CERT Insider Threat dataset...")
        # Load only first 5000 rows to save memory and speed up
        df = pd.read_csv(filepath, low_memory=False, nrows=5000)
        print(f"  Loaded {len(df)} rows from CERT Insider")
        
        # Parse different log types if needed
        # Assuming combined format with user activity
        
        # Key behavioral features
        behavior_features = [
            'user', 'date', 'device', 'activity',
            'file_tree', 'from_removable', 'to_removable',
            'content', 'size'
        ]
        
        # Use available columns
        available_cols = [col for col in behavior_features if col in df.columns]
        df = df[available_cols].copy()
        
        # Feature engineering for insider threats
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_after_hours'] = df['hour'].apply(
                lambda x: 1 if pd.notna(x) and (x < 7 or x > 18) else 0
            )
        
        # Aggregate user behavior patterns - handle missing columns
        agg_dict = {'activity': 'count'}
        if 'size' in df.columns:
            agg_dict['size'] = 'sum'
        
        if 'user' in df.columns:
            user_activity = df.groupby('user').agg(agg_dict).reset_index()
            
            if 'size' in agg_dict:
                user_activity.columns = ['user', 'activity_count', 'total_data_transferred']
            else:
                user_activity.columns = ['user', 'activity_count']
                user_activity['total_data_transferred'] = 0
            
            df = df.merge(user_activity, on='user', how='left')
        else:
            df['activity_count'] = 0
            df['total_data_transferred'] = 0
        
        # Insider risk score (can be enhanced with labeled data)
        df['insider_risk'] = 0  # Default to benign
        
        # Fill any NaN values
        df = df.fillna(0)
        
        return df
    
    def create_unified_features(self, unsw_df, cicids_df, cert_df):
        """Create unified feature set for ZTPO"""
        print("Creating unified feature representation...")
        
        unified_records = []
        
        # Process UNSW-NB15 records (sample to reduce memory)
        sample_size = min(2000, len(unsw_df))
        unsw_sample = unsw_df.sample(n=sample_size, random_state=42) if len(unsw_df) > sample_size else unsw_df
        
        for _, row in unsw_sample.iterrows():
            record = {
                'source': 'network',
                'timestamp': row.get('stime', 0),
                'threat_level': row.get('threat_score', 0),
                'context_type': 'external_traffic',
                'behavioral_anomaly': 0,
                'network_anomaly': row.get('threat_score', 0),
                'insider_risk': 0,
                'access_decision': 'deny' if row.get('threat_score', 0) > 0 else 'allow'
            }
            unified_records.append(record)
        
        # Process CICIDS2017 records (sample to reduce memory)
        sample_size = min(2000, len(cicids_df))
        cicids_sample = cicids_df.sample(n=sample_size, random_state=42) if len(cicids_df) > sample_size else cicids_df
        
        for _, row in cicids_sample.iterrows():
            record = {
                'source': 'context',
                'timestamp': 0,
                'threat_level': row.get('behavior_anomaly', 0),
                'context_type': 'flow_behavior',
                'behavioral_anomaly': row.get('behavior_anomaly', 0),
                'network_anomaly': row.get('behavior_anomaly', 0),
                'insider_risk': 0,
                'access_decision': 'deny' if row.get('behavior_anomaly', 0) > 0 else 'allow'
            }
            unified_records.append(record)
        
        # Process CERT Insider records (sample to reduce memory)
        sample_size = min(2000, len(cert_df))
        cert_sample = cert_df.sample(n=sample_size, random_state=42) if len(cert_df) > sample_size else cert_df
        
        for _, row in cert_sample.iterrows():
            record = {
                'source': 'behavioral',
                'timestamp': row.get('hour', 12),
                'threat_level': row.get('insider_risk', 0),
                'context_type': 'user_activity',
                'behavioral_anomaly': row.get('insider_risk', 0),
                'network_anomaly': 0,
                'insider_risk': row.get('insider_risk', 0),
                'access_decision': 'deny' if row.get('insider_risk', 0) > 0 else 'allow'
            }
            unified_records.append(record)
        
        print(f"  Created {len(unified_records)} unified records")
        
        unified_df = pd.DataFrame(unified_records)
        
        # Encode categorical features
        for col in ['source', 'context_type', 'access_decision']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            unified_df[col + '_encoded'] = self.label_encoders[col].fit_transform(
                unified_df[col]
            )
        
        return unified_df
    
    def prepare_state_action_pairs(self, unified_df):
        """Prepare state-action pairs for RL training"""
        print("Preparing state-action pairs...")
        
        # State features
        state_features = [
            'timestamp', 'threat_level', 'source_encoded',
            'context_type_encoded', 'behavioral_anomaly',
            'network_anomaly', 'insider_risk'
        ]
        
        X = unified_df[state_features].values
        
        # Actions: 0=allow, 1=restrict, 2=deny
        action_mapping = {'allow': 0, 'restrict': 1, 'deny': 2}
        y = unified_df['access_decision'].map(action_mapping).values
        
        # Normalize state features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

# ==================== DQN AGENT ====================

class DQNAgent:
    """Deep Q-Learning Agent for Zero Trust Policy Optimization"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        
        # Build neural networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Build Deep Q-Network"""
        model = keras.Sequential([
            keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """Train on batch from memory"""
        if len(self.memory) < self.batch_size:
            return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        # Q-learning update
        targets = rewards + self.gamma * (
            np.amax(self.target_model.predict_on_batch(next_states), axis=1)
        ) * (1 - dones)
        
        targets_full = self.model.predict_on_batch(states)
        
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
        
        loss = self.model.train_on_batch(states, targets_full)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def save(self, filepath):
        """Save model weights"""
        self.model.save_weights(filepath)
    
    def load(self, filepath):
        """Load model weights"""
        self.model.load_weights(filepath)

# ==================== ENVIRONMENT SIMULATOR ====================

class ZeroTrustEnvironment:
    """Simulates Zero Trust policy decisions"""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.current_idx = 0
        self.total_steps = len(data)
    
    def reset(self):
        """Reset environment"""
        self.current_idx = 0
        return self.data[self.current_idx]
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Get true label
        true_action = self.labels[self.current_idx]
        
        # Calculate reward
        reward = self._calculate_reward(action, true_action)
        
        # Move to next state
        self.current_idx += 1
        done = self.current_idx >= self.total_steps
        
        if not done:
            next_state = self.data[self.current_idx]
        else:
            next_state = self.data[self.current_idx - 1]
        
        return next_state, reward, done
    
    def _calculate_reward(self, action, true_action):
        """
        Reward function for ZTPO:
        - Correct decisions: +10
        - False positives (deny when should allow): -5
        - False negatives (allow when should deny): -20
        """
        if action == true_action:
            return 10
        elif action > true_action:  # Too restrictive
            return -5
        else:  # Too permissive (security risk)
            return -20

# ==================== TRAINING PIPELINE ====================

def train_ztpo_model(unsw_path, cicids_path, cert_path, episodes=20):
    """Main training pipeline"""
    
    print("="*60)
    print("Zero Trust Policy Optimization - Training Pipeline")
    print("="*60)
    
    # 1. Load and integrate datasets
    integrator = DatasetIntegrator()
    
    unsw_df = integrator.load_unsw_nb15(unsw_path)
    cicids_df = integrator.load_cicids2017(cicids_path)
    cert_df = integrator.load_cert_insider(cert_path)
    
    # 2. Create unified features
    unified_df = integrator.create_unified_features(unsw_df, cicids_df, cert_df)
    
    # 3. Prepare training data
    X, y = integrator.prepare_state_action_pairs(unified_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # 4. Initialize DQN agent
    state_size = X_train.shape[1]
    action_size = 3  # allow, restrict, deny
    
    agent = DQNAgent(state_size, action_size)
    env = ZeroTrustEnvironment(X_train, y_train)
    
    # 5. Training loop (OPTIMIZED)
    print("\n" + "="*60)
    print("Starting DQN Training...")
    print("="*60)
    
    scores = []
    losses = []
    
    # Use smaller batches per episode for faster training
    batch_episodes = 500  # Process 500 samples per episode instead of all 12000
    
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        total_reward = 0
        episode_loss = []
        
        # Only process a subset of data per episode
        for time_step in range(min(batch_episodes, len(X_train))):
            # Choose action
            action = agent.act(state)
            
            # Execute action
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Train agent every 4 steps instead of every step
            if time_step % 4 == 0:
                loss = agent.replay()
                if loss:
                    episode_loss.append(loss)
            
            if done:
                env.reset()
                state = env.data[env.current_idx]
                state = np.reshape(state, [1, state_size])
        
        # Update target network
        if e % 5 == 0:
            agent.update_target_model()
        
        scores.append(total_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
        
        # Print progress every episode
        print(f"Episode {e+1}/{episodes} | Reward: {total_reward:.0f} | Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f}")
    
    # 6. Evaluation
    print("\n" + "="*60)
    print("Evaluating Model...")
    print("="*60)
    
    test_env = ZeroTrustEnvironment(X_test, y_test)
    state = test_env.reset()
    state = np.reshape(state, [1, state_size])
    
    correct_decisions = 0
    false_positives = 0
    false_negatives = 0
    
    agent.epsilon = 0  # Disable exploration for evaluation
    
    for _ in range(len(X_test)):
        action = agent.act(state)
        true_action = y_test[test_env.current_idx]
        
        if action == true_action:
            correct_decisions += 1
        elif action > true_action:
            false_positives += 1
        else:
            false_negatives += 1
        
        next_state, _, done = test_env.step(action)
        state = np.reshape(next_state, [1, state_size])
        
        if done:
            break
    
    # Calculate metrics
    accuracy = (correct_decisions / len(X_test)) * 100
    fpr = (false_positives / len(X_test)) * 100
    fnr = (false_negatives / len(X_test)) * 100
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  False Positive Rate: {fpr:.2f}%")
    print(f"  False Negative Rate: {fnr:.2f}%")
    print(f"  Total Test Samples: {len(X_test)}")
    
    # 7. Save model and preprocessors
    print("\nSaving model and preprocessors...")
    
    # Create models directory if it doesn't exist
    save_dir = '/home/kali/ZTPO_Project/models/'
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'ztpo_model.weights.h5')
    scaler_path = os.path.join(save_dir, 'ztpo_scaler.pkl')
    encoders_path = os.path.join(save_dir, 'ztpo_label_encoders.pkl')
    
    agent.save(model_path)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(integrator.scaler, f)
    
    with open(encoders_path, 'wb') as f:
        pickle.dump(integrator.label_encoders, f)
    
    print("\nTraining Complete!")
    print("="*60)
    print("Saved files:")
    print(f"  - {model_path}")
    print(f"  - {scaler_path}")
    print(f"  - {encoders_path}")
    print("="*60)
    
    return agent, integrator, scores, losses

# ==================== USAGE ====================

if __name__ == "__main__":
    # Example usage
    print("\nTo train the model, call:")
    print("agent, integrator, scores, losses = train_ztpo_model(")
    print("    unsw_path='path/to/UNSW_NB15.csv',")
    print("    cicids_path='path/to/CICIDS2017.csv',")
    print("    cert_path='path/to/CERT_insider.csv',")
    print("    episodes=100")
    print(")")