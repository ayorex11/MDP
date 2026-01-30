from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import os
from werkzeug.utils import secure_filename
from ml_model import NSLKDDModel
from models import db, User, Analysis, Prediction, UserStatistics
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
        """Value Iteration Algorithm"""
        iteration = 0
        while True:
            delta = 0
            V_old = self.V.copy()
            
            for state in self.states:
                q_values = {}
                for action in self.actions:
                    q_value = 0
                    transitions = self.transitions.get((state, action), {})
                    
                    for next_state, prob in transitions.items():
                        reward = self.get_reward(state, action, next_state)
                        q_value += prob * (reward + self.gamma * V_old[next_state])
                    
                    q_values[action] = q_value
                    self.Q[state][action] = q_value
                
                if q_values:
                    self.V[state] = max(q_values.values())
                    delta = max(delta, abs(self.V[state] - V_old[state]))
            
            iteration += 1
            
            if delta < self.threshold:
                print(f"Value Iteration converged in {iteration} iterations")
                break
        
        # Extract optimal policy
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
    else:
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
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

def generate_attack_distribution_chart(attack_counts):
    """Generate attack type distribution pie chart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#4CAF50', '#F44336', '#FF9800', '#2196F3', '#9C27B0']
    ax.pie(attack_counts.values(), labels=attack_counts.keys(), autopct='%1.1f%%',
           colors=colors[:len(attack_counts)], startangle=90)
    ax.set_title('Attack Type Distribution', fontsize=14, fontweight='bold')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    chart_image = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return chart_image

# Initialize and train MDP model at startup
print("Training MDP model...")
mdp = MDPCyberDefense(gamma=0.95, threshold=0.01)
mdp.value_iteration()
print("MDP model trained successfully!")

# Load ML models
print("\nLoading NSL-KDD ML models...")
ml_model = NSLKDDModel()
try:
    ml_model.load_models()
    print("ML models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load ML models: {e}")
    print("ML features will be disabled. Run 'python ml_model.py' to train models.")
    ml_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ===== Authentication Routes =====

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
        
        # Strong password validation
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return redirect(url_for('register'))
        
        import re
        if not re.search(r'[a-z]', password):
            flash('Password must contain at least one lowercase letter.', 'danger')
            return redirect(url_for('register'))
        
        if not re.search(r'[A-Z]', password):
            flash('Password must contain at least one uppercase letter.', 'danger')
            return redirect(url_for('register'))
        
        if not re.search(r'[0-9]', password):
            flash('Password must contain at least one number.', 'danger')
            return redirect(url_for('register'))
        
        if not re.search(r'[^a-zA-Z0-9]', password):
            flash('Password must contain at least one special character.', 'danger')
            return redirect(url_for('register'))
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        # Create user statistics
        stats = UserStatistics(user=user)
        
        db.session.add(user)
        db.session.add(stats)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'yes'
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f'Welcome back, {user.username}!', 'success')
            
            # Redirect to next page or index
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username/email or password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    stats = current_user.statistics
    analysis_count = Analysis.query.filter_by(user_id=current_user.id).count()
    recent_analyses = Analysis.query.filter_by(user_id=current_user.id)\
        .order_by(Analysis.upload_timestamp.desc()).limit(5).all()
    
    return render_template('profile.html',
                         stats=stats,
                         analysis_count=analysis_count,
                         recent_analyses=recent_analyses)

@app.route('/history')
@login_required
def history():
    """Analysis history page"""
    analyses = Analysis.query.filter_by(user_id=current_user.id)\
        .order_by(Analysis.upload_timestamp.desc()).all()
    
    total_records = sum(a.total_records for a in analyses)
    
    return render_template('history.html',
                         analyses=analyses,
                         total_records=total_records)

@app.route('/analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    """View specific analysis details"""
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Ensure user owns this analysis
    if analysis.user_id != current_user.id:
        flash('You do not have permission to view this analysis.', 'danger')
        return redirect(url_for('history'))
    
    predictions = Prediction.query.filter_by(analysis_id=analysis_id).all()
    
    # Calculate attack counts
    attack_counts = {}
    for pred in predictions:
        attack_counts[pred.attack_type] = attack_counts.get(pred.attack_type, 0) + 1
    
    # Generate chart
    chart_image = generate_attack_distribution_chart(attack_counts)
    
    return render_template('view_analysis.html',
                         analysis=analysis,
                         predictions=predictions,
                         attack_counts=attack_counts,
                         chart_image=chart_image)

# ===== Main Application Routes =====

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict')
@login_required
def predict():
    """Prediction form page"""
    return render_template('predict.html')

@app.route('/results', methods=['POST'])
@login_required
def results():
    """Results page with MDP analysis"""
    try:
        traffic = int(request.form.get('traffic', 500))
        suspicious_indicators = int(request.form.get('suspicious_indicators', 0))
        
        current_state = determine_state(traffic, suspicious_indicators)
        prediction = mdp.predict(current_state)
        risk_level = calculate_risk_level(prediction['next_states'])
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
        flash('An error occurred during analysis.', 'danger')
        return redirect(url_for('predict'))

@app.route('/upload')
@login_required
def upload():
    """File upload page"""
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    """Analyze uploaded NSL-KDD data"""
    if ml_model is None:
        flash('ML models not loaded. Please contact administrator.', 'danger')
        return redirect(url_for('upload'))
    
    try:
        if 'file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(url_for('upload'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(url_for('upload'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and analyze data
            df = ml_model.load_nslkdd_data(filepath)
            
            # Limit to first 1000 records for performance
            if len(df) > 1000:
                df = df.head(1000)
            
            # Get predictions
            predictions = ml_model.predict_from_features(df)
            
            # Create analysis record
            analysis = Analysis(
                user_id=current_user.id,
                filename=filename,
                total_records=len(df)
            )
            db.session.add(analysis)
            db.session.flush()  # Get analysis ID
            
            # Store predictions and prepare results
            results = []
            prediction_records = []
            
            for i in range(len(df)):
                state = predictions['states'][i]
                attack_type = predictions['attack_types'][i]
                mdp_pred = mdp.predict(state)
                
                # Create prediction record
                pred_record = Prediction(
                    analysis_id=analysis.id,
                    record_index=i + 1,
                    mdp_state=state,
                    attack_type=attack_type,
                    recommended_action=mdp_pred['optimal_action'],
                    state_value=round(mdp_pred['state_value'], 2)
                )
                prediction_records.append(pred_record)
                
                results.append({
                    'index': i + 1,
                    'state': state,
                    'attack_type': attack_type,
                    'recommended_action': mdp_pred['optimal_action'],
                    'state_value': round(mdp_pred['state_value'], 2)
                })
            
            # Bulk insert predictions
            db.session.bulk_save_objects(prediction_records)
            
            # Update user statistics
            if not current_user.statistics:
                stats = UserStatistics(user_id=current_user.id)
                db.session.add(stats)
            else:
                stats = current_user.statistics
            
            stats.update_statistics(results)
            
            db.session.commit()
            
            # Generate attack type distribution chart
            attack_counts = {}
            for r in results:
                attack_counts[r['attack_type']] = attack_counts.get(r['attack_type'], 0) + 1
            
            chart_image = generate_attack_distribution_chart(attack_counts)
            
            flash(f'Successfully analyzed {len(results)} records!', 'success')
            
            return render_template(
                'analyze.html',
                results=results[:100],
                total_records=len(results),
                attack_counts=attack_counts,
                chart_image=chart_image
            )
            
    except Exception as e:
        print(f"Error in analyze route: {e}")
        import traceback
        traceback.print_exc()
        db.session.rollback()
        flash(f'Error analyzing file: {str(e)}', 'danger')
        return redirect(url_for('upload'))

@app.route('/statistics')
@login_required
def statistics():
    """Statistics dashboard"""
    stats = current_user.statistics
    
    if not stats or stats.total_analyzed == 0:
        return render_template('statistics.html', no_data=True)
    
    # Generate visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Attack type distribution
    if stats.attack_counts:
        colors = ['#4CAF50', '#F44336', '#FF9800', '#2196F3', '#9C27B0', '#795548']
        ax1.pie(stats.attack_counts.values(), 
                labels=stats.attack_counts.keys(),
                autopct='%1.1f%%', colors=colors[:len(stats.attack_counts)],
                startangle=90)
        ax1.set_title('Attack Type Distribution', fontsize=12, fontweight='bold')
    
    # State distribution
    if stats.state_counts:
        states = list(stats.state_counts.keys())
        counts = list(stats.state_counts.values())
        ax2.bar(states, counts, color='#667eea', alpha=0.8)
        ax2.set_title('MDP State Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('State')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
    
    # Recommended actions
    if stats.action_counts:
        actions = list(stats.action_counts.keys())
        counts = list(stats.action_counts.values())
        ax3.bar(actions, counts, color='#764ba2', alpha=0.8)
        ax3.set_title('Recommended Actions Frequency', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Action')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
    
    # Attack detection rate
    total = stats.total_analyzed
    normal_count = stats.attack_counts.get('Normal', 0)
    attack_count = total - normal_count
    
    ax4.pie([normal_count, attack_count], labels=['Normal', 'Attack'],
            autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
    ax4.set_title('Attack Detection Rate', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    stats_chart = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return render_template(
        'statistics.html',
        total_analyzed=stats.total_analyzed,
        attack_counts=stats.attack_counts,
        state_counts=stats.state_counts,
        action_counts=stats.action_counts,
        stats_chart=stats_chart,
        no_data=False
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
