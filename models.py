"""
Database models for MDP Cyber-Attack Predictor
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User account model"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    statistics = db.relationship('UserStatistics', backref='user', uselist=False, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Analysis(db.Model):
    """Analysis session model - stores metadata about uploaded files"""
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    total_records = db.Column(db.Integer, nullable=False)
    upload_timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='analysis', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Analysis {self.filename} by User {self.user_id}>'


class Prediction(db.Model):
    """Individual prediction result"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id'), nullable=False, index=True)
    record_index = db.Column(db.Integer, nullable=False)
    mdp_state = db.Column(db.String(50), nullable=False, index=True)
    attack_type = db.Column(db.String(50), nullable=False, index=True)
    recommended_action = db.Column(db.String(50), nullable=False)
    state_value = db.Column(db.Float, nullable=False)
    confidence_score = db.Column(db.Float)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.attack_type}>'


class UserStatistics(db.Model):
    """Aggregated statistics per user"""
    __tablename__ = 'user_statistics'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True, index=True)
    total_analyzed = db.Column(db.Integer, default=0)
    attack_counts_json = db.Column(db.Text, default='{}')
    state_counts_json = db.Column(db.Text, default='{}')
    action_counts_json = db.Column(db.Text, default='{}')
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @property
    def attack_counts(self):
        """Get attack counts as dictionary"""
        return json.loads(self.attack_counts_json) if self.attack_counts_json else {}
    
    @attack_counts.setter
    def attack_counts(self, value):
        """Set attack counts from dictionary"""
        self.attack_counts_json = json.dumps(value)
    
    @property
    def state_counts(self):
        """Get state counts as dictionary"""
        return json.loads(self.state_counts_json) if self.state_counts_json else {}
    
    @state_counts.setter
    def state_counts(self, value):
        """Set state counts from dictionary"""
        self.state_counts_json = json.dumps(value)
    
    @property
    def action_counts(self):
        """Get action counts as dictionary"""
        return json.loads(self.action_counts_json) if self.action_counts_json else {}
    
    @action_counts.setter
    def action_counts(self, value):
        """Set action counts from dictionary"""
        self.action_counts_json = json.dumps(value)
    
    def update_statistics(self, predictions):
        """Update statistics from a list of predictions"""
        attack_counts = self.attack_counts
        state_counts = self.state_counts
        action_counts = self.action_counts
        
        for pred in predictions:
            # Update attack counts
            attack_counts[pred['attack_type']] = attack_counts.get(pred['attack_type'], 0) + 1
            
            # Update state counts
            state_counts[pred['state']] = state_counts.get(pred['state'], 0) + 1
            
            # Update action counts
            action_counts[pred['recommended_action']] = action_counts.get(pred['recommended_action'], 0) + 1
        
        self.attack_counts = attack_counts
        self.state_counts = state_counts
        self.action_counts = action_counts
        self.total_analyzed += len(predictions)
        self.last_updated = datetime.utcnow()
    
    def __repr__(self):
        return f'<UserStatistics for User {self.user_id}>'
