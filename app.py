from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

class MDPCyberDefense:
    """
    Markov Decision Process for Cyber-Attack Prediction
    Uses Value Iteration algorithm to compute optimal defense policies
    """
    
    def __init__(self, gamma=0.95, threshold=0.01):
        self.gamma = gamma  # Discount factor
        self.threshold = threshold  # Convergence threshold
        
        # Define states
        self.states = [
            'Normal',
            'Low_Suspicious',
            'High_Suspicious',
            'Attack_Detected',
            'Attack_Blocked',
            'System_Crashed'
        ]
        
        # Define actions
        self.actions = [
            'No_Action',
            'Increase_Monitor',
            'Rate_Limit',
            'Block_IPs',
            'Emergency_Stop'
        ]
        
        # State rewards
        self.state_rewards = {
            'Normal': 10,
            'Low_Suspicious': 0,
            'High_Suspicious': -20,
            'Attack_Detected': -50,
            'Attack_Blocked': 200,
            'System_Crashed': -500
        }
        
        # Action costs
        self.action_costs = {
            'No_Action': 0,
            'Increase_Monitor': -5,
            'Rate_Limit': -15,
            'Block_IPs': -30,
            'Emergency_Stop': -100
        }
        
        # Initialize transition probabilities
        self.transitions = self._build_transitions()
        
        # Initialize value function and policy
        self.V = {s: 0.0 for s in self.states}
        self.policy = {s: None for s in self.states}
        self.Q = {s: {a: 0.0 for a in self.actions} for s in self.states}
        
    def _build_transitions(self):
        """Build transition probability matrix P(s'|s,a)"""
        T = {}
        
        # Normal state transitions
        T[('Normal', 'No_Action')] = {
            'Normal': 0.90,
            'Low_Suspicious': 0.10
        }
        T[('Normal', 'Increase_Monitor')] = {
            'Normal': 0.95,
            'Low_Suspicious': 0.05
        }
        T[('Normal', 'Rate_Limit')] = {
            'Normal': 0.98,
            'Low_Suspicious': 0.02
        }
        T[('Normal', 'Block_IPs')] = {
            'Normal': 0.99,
            'Low_Suspicious': 0.01
        }
        T[('Normal', 'Emergency_Stop')] = {
            'Normal': 1.0
        }
        
        # Low_Suspicious state transitions
        T[('Low_Suspicious', 'No_Action')] = {
            'Normal': 0.30,
            'Low_Suspicious': 0.40,
            'High_Suspicious': 0.30
        }
        T[('Low_Suspicious', 'Increase_Monitor')] = {
            'Normal': 0.50,
            'Low_Suspicious': 0.35,
            'High_Suspicious': 0.15
        }
        T[('Low_Suspicious', 'Rate_Limit')] = {
            'Normal': 0.60,
            'Low_Suspicious': 0.30,
            'High_Suspicious': 0.10
        }
        T[('Low_Suspicious', 'Block_IPs')] = {
            'Normal': 0.70,
            'Low_Suspicious': 0.20,
            'Attack_Blocked': 0.10
        }
        T[('Low_Suspicious', 'Emergency_Stop')] = {
            'Normal': 0.80,
            'Low_Suspicious': 0.15,
            'Attack_Blocked': 0.05
        }
        
        # High_Suspicious state transitions
        T[('High_Suspicious', 'No_Action')] = {
            'High_Suspicious': 0.20,
            'Attack_Detected': 0.50,
            'System_Crashed': 0.30
        }
        T[('High_Suspicious', 'Increase_Monitor')] = {
            'Low_Suspicious': 0.20,
            'High_Suspicious': 0.30,
            'Attack_Detected': 0.40,
            'System_Crashed': 0.10
        }
        T[('High_Suspicious', 'Rate_Limit')] = {
            'Low_Suspicious': 0.25,
            'High_Suspicious': 0.15,
            'Attack_Detected': 0.35,
            'Attack_Blocked': 0.15,
            'System_Crashed': 0.10
        }
        T[('High_Suspicious', 'Block_IPs')] = {
            'Low_Suspicious': 0.10,
            'Attack_Detected': 0.20,
            'Attack_Blocked': 0.60,
            'System_Crashed': 0.10
        }
        T[('High_Suspicious', 'Emergency_Stop')] = {
            'Low_Suspicious': 0.05,
            'Attack_Blocked': 0.85,
            'System_Crashed': 0.10
        }
        
        # Attack_Detected state transitions
        T[('Attack_Detected', 'No_Action')] = {
            'Attack_Detected': 0.30,
            'System_Crashed': 0.70
        }
        T[('Attack_Detected', 'Increase_Monitor')] = {
            'Attack_Detected': 0.50,
            'Attack_Blocked': 0.20,
            'System_Crashed': 0.30
        }
        T[('Attack_Detected', 'Rate_Limit')] = {
            'Attack_Detected': 0.30,
            'Attack_Blocked': 0.50,
            'System_Crashed': 0.20
        }
        T[('Attack_Detected', 'Block_IPs')] = {
            'Attack_Detected': 0.15,
            'Attack_Blocked': 0.75,
            'System_Crashed': 0.10
        }
        T[('Attack_Detected', 'Emergency_Stop')] = {
            'Attack_Blocked': 0.90,
            'System_Crashed': 0.10
        }
        
        # Terminal states (absorbing)
        for action in self.actions:
            T[('Attack_Blocked', action)] = {'Attack_Blocked': 1.0}
            T[('System_Crashed', action)] = {'System_Crashed': 1.0}
        
        return T
    
    def get_reward(self, state, action, next_state):
        """Calculate reward R(s,a,s')"""
        return self.state_rewards[next_state] + self.action_costs[action]
    
    def value_iteration(self):
        """
        Value Iteration Algorithm:
        1. Initialize V(s) = 0 for all states
        2. Repeat until convergence:
            - For each state s:
                - For each action a:
                    - Q(s,a) = Σ P(s'|s,a) × (R(s,a,s') + γV(s'))
                - V(s) = max_a Q(s,a)
        3. Extract policy: π(s) = argmax_a Q(s,a)
        """
        iteration = 0
        while True:
            delta = 0
            V_old = self.V.copy()
            
            # Update value for each state
            for state in self.states:
                # Compute Q-values for all actions
                q_values = {}
                for action in self.actions:
                    q_value = 0
                    
                    # Get transition probabilities for this (state, action) pair
                    transitions = self.transitions.get((state, action), {})
                    
                    # Q(s,a) = Σ P(s'|s,a) × (R(s,a,s') + γV(s'))
                    for next_state, prob in transitions.items():
                        reward = self.get_reward(state, action, next_state)
                        q_value += prob * (reward + self.gamma * V_old[next_state])
                    
                    q_values[action] = q_value
                    self.Q[state][action] = q_value
                
                # V(s) = max_a Q(s,a)
                if q_values:
                    self.V[state] = max(q_values.values())
                    delta = max(delta, abs(self.V[state] - V_old[state]))
            
            iteration += 1
            
            # Check convergence
            if delta < self.threshold:
                print(f"Value Iteration converged in {iteration} iterations")
                break
        
        # Extract optimal policy: π(s) = argmax_a Q(s,a)
        for state in self.states:
            self.policy[state] = max(self.Q[state], key=self.Q[state].get)
    
    def get_next_state_probabilities(self, state, action):
        """Get probability distribution over next states"""
        return self.transitions.get((state, action), {})
    
    def predict(self, state):
        """Get optimal action and predictions for a given state"""
        optimal_action = self.policy[state]
        next_states = self.get_next_state_probabilities(state, optimal_action)
        state_value = self.V[state]
        q_values = self.Q[state]
        
        return {
            'current_state': state,
            'optimal_action': optimal_action,
            'next_states': next_states,
            'state_value': state_value,
            'q_values': q_values
        }

def determine_state(traffic, suspicious_indicators):
    """Determine current state based on network metrics"""
    if suspicious_indicators == 0 and traffic < 1000:
        return 'Normal'
    elif suspicious_indicators == 0 and traffic >= 1000:
        return 'Low_Suspicious'
    elif suspicious_indicators < 5:
        return 'Low_Suspicious'
    elif suspicious_indicators < 10:
        return 'High_Suspicious'
    else:  # suspicious_indicators >= 10
        return 'Attack_Detected'

def calculate_risk_level(next_states):
    """Calculate risk level based on predicted next states"""
    high_risk_prob = next_states.get('Attack_Detected', 0) + next_states.get('High_Suspicious', 0)
    
    if high_risk_prob > 0.5:
        return 'HIGH'
    elif high_risk_prob > 0.2:
        return 'MEDIUM'
    else:
        return 'LOW'

def generate_charts(next_states, q_values, optimal_action):
    """Generate visualization charts as base64 encoded images"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chart 1: Predicted Next States
    if next_states:
        states = list(next_states.keys())
        probs = list(next_states.values())
        colors = ['#667eea' if p == max(probs) else '#764ba2' for p in probs]
        
        ax1.bar(states, probs, color=colors, alpha=0.8)
        ax1.set_xlabel('Next State', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Probability', fontsize=10, fontweight='bold')
        ax1.set_title('Predicted Next States', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
    
    # Chart 2: Q-Values for All Actions
    actions = list(q_values.keys())
    values = list(q_values.values())
    colors = ['#4CAF50' if action == optimal_action else '#764ba2' for action in actions]
    
    ax2.bar(actions, values, color=colors, alpha=0.8)
    ax2.set_xlabel('Action', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Q-Value', fontsize=10, fontweight='bold')
    ax2.set_title('Q-Values (Optimal in Green)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

# Initialize and train MDP model at startup
print("Training MDP model...")
mdp = MDPCyberDefense(gamma=0.95, threshold=0.01)
mdp.value_iteration()
print("MDP model trained successfully!")
print(f"State values: {mdp.V}")
print(f"Optimal policy: {mdp.policy}")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict')
def predict():
    """Prediction form page"""
    return render_template('predict.html')

@app.route('/results', methods=['POST'])
def results():
    """Results page with MDP analysis"""
    try:
        # Get form data
        traffic = int(request.form.get('traffic', 500))
        suspicious_indicators = int(request.form.get('suspicious_indicators', 0))
        
        # Determine current state
        current_state = determine_state(traffic, suspicious_indicators)
        
        # Get MDP prediction
        prediction = mdp.predict(current_state)
        
        # Calculate risk level
        risk_level = calculate_risk_level(prediction['next_states'])
        
        # Generate charts
        chart_image = generate_charts(
            prediction['next_states'],
            prediction['q_values'],
            prediction['optimal_action']
        )
        
        return render_template(
            'results.html',
            traffic=traffic,
            suspicious_indicators=suspicious_indicators,
            current_state=current_state,
            optimal_action=prediction['optimal_action'],
            state_value=round(prediction['state_value'], 2),
            risk_level=risk_level,
            next_states=prediction['next_states'],
            chart_image=chart_image
        )
    except Exception as e:
        print(f"Error in results route: {e}")
        return redirect(url_for('predict'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
