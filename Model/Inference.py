"""
Zero Trust Policy Optimization - Inference Script
Use trained model to make real-time access control decisions
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

class ZTPOInference:
    """Real-time Zero Trust policy decision engine"""
    
    def __init__(self, model_path, scaler_path, encoders_path):
        """
        Load trained model and preprocessors
        
        Args:
            model_path: Path to trained model weights
            scaler_path: Path to feature scaler
            encoders_path: Path to label encoders
        """
        print("Loading ZTPO model...")
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load label encoders
        with open(encoders_path, 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        # Rebuild and load model
        state_size = 7  # Number of features
        action_size = 3  # allow, restrict, deny
        
        self.model = keras.Sequential([
            keras.layers.Dense(128, input_dim=state_size, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(action_size, activation='linear')
        ])
        
        self.model.load_weights(model_path)
        
        self.action_map = {0: 'allow', 1: 'restrict', 2: 'deny'}
        
        print("✅ Model loaded successfully!")
    
    def predict_access_decision(self, 
                               source='network',
                               context_type='external_traffic',
                               timestamp=12,
                               threat_level=0.0,
                               behavioral_anomaly=0.0,
                               network_anomaly=0.0,
                               insider_risk=0.0):
        """
        Predict access control decision based on context
        
        Args:
            source: 'network', 'context', or 'behavioral'
            context_type: Type of access context
            timestamp: Hour of access (0-23)
            threat_level: Overall threat score (0-1)
            behavioral_anomaly: Behavioral anomaly score (0-1)
            network_anomaly: Network anomaly score (0-1)
            insider_risk: Insider threat risk score (0-1)
        
        Returns:
            dict: Decision and confidence scores
        """
        
        # Encode categorical features
        source_encoded = self.label_encoders['source'].transform([source])[0]
        context_encoded = self.label_encoders['context_type'].transform([context_type])[0]
        
        # Create feature vector
        features = np.array([[
            timestamp,
            threat_level,
            source_encoded,
            context_encoded,
            behavioral_anomaly,
            network_anomaly,
            insider_risk
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        q_values = self.model.predict(features_scaled, verbose=0)[0]
        action_idx = np.argmax(q_values)
        decision = self.action_map[action_idx]
        
        # Calculate confidence scores
        confidence_scores = {
            'allow': float(q_values[0]),
            'restrict': float(q_values[1]),
            'deny': float(q_values[2])
        }
        
        # Normalize to percentages
        total = sum(abs(v) for v in confidence_scores.values())
        if total > 0:
            confidence_percentages = {
                k: abs(v) / total * 100 for k, v in confidence_scores.items()
            }
        else:
            confidence_percentages = confidence_scores
        
        return {
            'decision': decision,
            'confidence': confidence_percentages[decision],
            'all_scores': confidence_percentages,
            'raw_q_values': q_values.tolist()
        }
    
    def batch_predict(self, access_requests):
        """
        Process multiple access requests at once
        
        Args:
            access_requests: List of dictionaries with request parameters
        
        Returns:
            List of decision dictionaries
        """
        results = []
        for request in access_requests:
            result = self.predict_access_decision(**request)
            results.append(result)
        return results
    
    def explain_decision(self, decision_result):
        """
        Provide human-readable explanation of decision
        
        Args:
            decision_result: Result from predict_access_decision
        
        Returns:
            String explanation
        """
        decision = decision_result['decision']
        confidence = decision_result['confidence']
        scores = decision_result['all_scores']
        
        explanation = f"""
╔═══════════════════════════════════════════════════════════╗
║           ZERO TRUST POLICY DECISION                      ║
╚═══════════════════════════════════════════════════════════╝

DECISION: {decision.upper()}
CONFIDENCE: {confidence:.1f}%

SCORE BREAKDOWN:
  • Allow:    {scores['allow']:.1f}%  {'█' * int(scores['allow']/5)}
  • Restrict: {scores['restrict']:.1f}%  {'█' * int(scores['restrict']/5)}
  • Deny:     {scores['deny']:.1f}%  {'█' * int(scores['deny']/5)}

RECOMMENDATION:
"""
        
        if decision == 'allow':
            explanation += "  ✅ Access GRANTED - Low risk context detected"
        elif decision == 'restrict':
            explanation += "  ⚠️  Access RESTRICTED - Elevated monitoring required"
        else:
            explanation += "  ❌ Access DENIED - High risk context detected"
        
        return explanation


# ==================== USAGE EXAMPLES ====================

def demo_scenarios():
    """Demonstrate various access scenarios"""
    
    # Initialize inference engine
    ztpo = ZTPOInference(
        model_path='ztpo_model.weights.h5',
        scaler_path='ztpo_scaler.pkl',
        encoders_path='ztpo_label_encoders.pkl'
    )
    
    print("\n" + "="*60)
    print("SCENARIO TESTING - Zero Trust Policy Optimization")
    print("="*60)
    
    # Scenario 1: Normal business hours access
    print("\n🔷 SCENARIO 1: Normal User - Business Hours")
    result1 = ztpo.predict_access_decision(
        source='behavioral',
        context_type='user_activity',
        timestamp=14,  # 2 PM
        threat_level=0.1,
        behavioral_anomaly=0.0,
        network_anomaly=0.0,
        insider_risk=0.0
    )
    print(ztpo.explain_decision(result1))
    
    # Scenario 2: After-hours access with anomaly
    print("\n🔷 SCENARIO 2: After-Hours Access with Behavioral Anomaly")
    result2 = ztpo.predict_access_decision(
        source='behavioral',
        context_type='user_activity',
        timestamp=23,  # 11 PM
        threat_level=0.5,
        behavioral_anomaly=0.7,
        network_anomaly=0.2,
        insider_risk=0.6
    )
    print(ztpo.explain_decision(result2))
    
    # Scenario 3: Network intrusion attempt
    print("\n🔷 SCENARIO 3: Detected Network Intrusion")
    result3 = ztpo.predict_access_decision(
        source='network',
        context_type='external_traffic',
        timestamp=10,
        threat_level=0.9,
        behavioral_anomaly=0.0,
        network_anomaly=0.95,
        insider_risk=0.0
    )
    print(ztpo.explain_decision(result3))
    
    # Scenario 4: Contextual flow analysis
    print("\n🔷 SCENARIO 4: Normal Network Flow")
    result4 = ztpo.predict_access_decision(
        source='context',
        context_type='flow_behavior',
        timestamp=9,
        threat_level=0.05,
        behavioral_anomaly=0.0,
        network_anomaly=0.0,
        insider_risk=0.0
    )
    print(ztpo.explain_decision(result4))
    
    # Batch processing example
    print("\n" + "="*60)
    print("BATCH PROCESSING EXAMPLE")
    print("="*60)
    
    batch_requests = [
        {'source': 'network', 'threat_level': 0.2, 'network_anomaly': 0.1},
        {'source': 'behavioral', 'threat_level': 0.8, 'insider_risk': 0.9},
        {'source': 'context', 'threat_level': 0.0, 'behavioral_anomaly': 0.0}
    ]
    
    batch_results = ztpo.batch_predict(batch_requests)
    
    for i, result in enumerate(batch_results, 1):
        print(f"\nRequest {i}: {result['decision'].upper()} "
              f"(Confidence: {result['confidence']:.1f}%)")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


def interactive_mode():
    """Interactive CLI for testing access decisions"""
    
    ztpo = ZTPOInference(
        model_path='ztpo_model.weights.h5',
        scaler_path='ztpo_scaler.pkl',
        encoders_path='ztpo_label_encoders.pkl'
    )
    
    print("\n" + "="*60)
    print("INTERACTIVE ZERO TRUST POLICY TESTER")
    print("="*60)
    print("Enter access context parameters (or 'quit' to exit)")
    
    while True:
        print("\n" + "-"*60)
        
        try:
            source = input("Source (network/context/behavioral) [network]: ").strip() or 'network'
            
            if source.lower() == 'quit':
                break
            
            threat = float(input("Threat level (0.0-1.0) [0.0]: ") or '0.0')
            behavior = float(input("Behavioral anomaly (0.0-1.0) [0.0]: ") or '0.0')
            network = float(input("Network anomaly (0.0-1.0) [0.0]: ") or '0.0')
            insider = float(input("Insider risk (0.0-1.0) [0.0]: ") or '0.0')
            
            result = ztpo.predict_access_decision(
                source=source,
                threat_level=threat,
                behavioral_anomaly=behavior,
                network_anomaly=network,
                insider_risk=insider
            )
            
            print(ztpo.explain_decision(result))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nExiting interactive mode...")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════╗
║   ZERO TRUST POLICY OPTIMIZATION - INFERENCE ENGINE       ║
║   AI-Driven Access Control Decision System                ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("\nChoose mode:")
    print("1. Run demo scenarios")
    print("2. Interactive testing mode")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        demo_scenarios()
    elif choice == '2':
        interactive_mode()
    else:
        print("Running demo scenarios by default...")
        demo_scenarios()