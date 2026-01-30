"""
ML Model for NSL-KDD Dataset Integration
Provides state classification and attack type detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class NSLKDDModel:
    """Machine Learning model for NSL-KDD intrusion detection"""
    
    def __init__(self):
        self.feature_names = self._load_feature_names()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.state_classifier = None
        self.attack_classifier = None
        self.is_trained = False
        
        # Attack type to MDP state mapping
        self.attack_to_state = {
            'normal': 'Normal',
            'neptune': 'High_Suspicious',  # DoS
            'smurf': 'High_Suspicious',     # DoS
            'pod': 'High_Suspicious',       # DoS
            'teardrop': 'High_Suspicious',  # DoS
            'land': 'High_Suspicious',      # DoS
            'back': 'High_Suspicious',      # DoS
            'apache2': 'High_Suspicious',   # DoS
            'udpstorm': 'High_Suspicious',  # DoS
            'processtable': 'High_Suspicious', # DoS
            'mailbomb': 'High_Suspicious',  # DoS
            'portsweep': 'Low_Suspicious',  # Probe
            'ipsweep': 'Low_Suspicious',    # Probe
            'nmap': 'Low_Suspicious',       # Probe
            'satan': 'Low_Suspicious',      # Probe
            'saint': 'Low_Suspicious',      # Probe
            'mscan': 'Low_Suspicious',      # Probe
            'guess_passwd': 'Attack_Detected',  # R2L
            'ftp_write': 'Attack_Detected',     # R2L
            'imap': 'Attack_Detected',          # R2L
            'phf': 'Attack_Detected',           # R2L
            'multihop': 'Attack_Detected',      # R2L
            'warezmaster': 'Attack_Detected',   # R2L
            'warezclient': 'Attack_Detected',   # R2L
            'spy': 'Attack_Detected',           # R2L
            'xlock': 'Attack_Detected',         # R2L
            'xsnoop': 'Attack_Detected',        # R2L
            'snmpguess': 'Attack_Detected',     # R2L
            'snmpgetattack': 'Attack_Detected', # R2L
            'httptunnel': 'Attack_Detected',    # R2L
            'sendmail': 'Attack_Detected',      # R2L
            'named': 'Attack_Detected',         # R2L
            'buffer_overflow': 'Attack_Detected', # U2R
            'loadmodule': 'Attack_Detected',      # U2R
            'rootkit': 'Attack_Detected',         # U2R
            'perl': 'Attack_Detected',            # U2R
            'sqlattack': 'Attack_Detected',       # U2R
            'xterm': 'Attack_Detected',           # U2R
            'ps': 'Attack_Detected',              # U2R
        }
        
        # Attack type categories
        self.attack_categories = {
            'normal': 'Normal',
            'neptune': 'DoS', 'smurf': 'DoS', 'pod': 'DoS', 'teardrop': 'DoS',
            'land': 'DoS', 'back': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
            'processtable': 'DoS', 'mailbomb': 'DoS',
            'portsweep': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe',
            'satan': 'Probe', 'saint': 'Probe', 'mscan': 'Probe',
            'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L',
            'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
            'warezclient': 'R2L', 'spy': 'R2L', 'xlock': 'R2L',
            'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
            'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
            'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
        }
    
    def _load_feature_names(self):
        """Load feature names from file"""
        feature_file = 'data/feature_names.txt'
        if os.path.exists(feature_file):
            with open(feature_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            # Default 41 features
            return [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate'
            ]
    
    def load_nslkdd_data(self, filepath):
        """Load NSL-KDD dataset from TXT file"""
        # Column names: 41 features + attack_type + difficulty
        columns = self.feature_names + ['attack_type', 'difficulty']
        
        # Load data
        df = pd.read_csv(filepath, names=columns, header=None)
        
        # Clean attack type (remove trailing dot if present)
        df['attack_type'] = df['attack_type'].str.lower().str.strip()
        
        return df
    
    def preprocess_features(self, df, fit=False):
        """Preprocess features for ML model"""
        df = df.copy()
        
        # Encode categorical features
        categorical_features = ['protocol_type', 'service', 'flag']
        
        for feature in categorical_features:
            if feature in df.columns:
                if fit:
                    self.label_encoders[feature] = LabelEncoder()
                    df[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    if feature in self.label_encoders:
                        # Handle unknown categories
                        df[feature] = df[feature].astype(str)
                        known_classes = set(self.label_encoders[feature].classes_)
                        df[feature] = df[feature].apply(lambda x: x if x in known_classes else 'unknown')
                        
                        # Add 'unknown' to encoder if not present
                        if 'unknown' not in self.label_encoders[feature].classes_:
                            self.label_encoders[feature].classes_ = np.append(
                                self.label_encoders[feature].classes_, 'unknown'
                            )
                        
                        df[feature] = self.label_encoders[feature].transform(df[feature])
        
        # Select only the 41 features
        X = df[self.feature_names].values
        
        # Scale features
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X
    
    def train(self, train_file='data/KDDTrain+.TXT', test_file='data/KDDTest+.TXT'):
        """Train both state and attack classifiers"""
        print("Loading NSL-KDD training data...")
        train_df = self.load_nslkdd_data(train_file)
        
        print(f"Training data loaded: {len(train_df)} records")
        print(f"Attack types: {train_df['attack_type'].nunique()}")
        
        # Create labels
        train_df['mdp_state'] = train_df['attack_type'].map(
            lambda x: self.attack_to_state.get(x, 'Attack_Detected')
        )
        train_df['attack_category'] = train_df['attack_type'].map(
            lambda x: self.attack_categories.get(x, 'Unknown')
        )
        
        # Preprocess features
        print("Preprocessing features...")
        X_train = self.preprocess_features(train_df, fit=True)
        y_state = train_df['mdp_state'].values
        y_attack = train_df['attack_category'].values
        
        # Train state classifier
        print("\nTraining MDP State Classifier...")
        self.state_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.state_classifier.fit(X_train, y_state)
        
        # Train attack type classifier
        print("Training Attack Type Classifier...")
        self.attack_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.attack_classifier.fit(X_train, y_attack)
        
        # Evaluate on test set if available
        if os.path.exists(test_file):
            print("\nEvaluating on test set...")
            test_df = self.load_nslkdd_data(test_file)
            test_df['mdp_state'] = test_df['attack_type'].map(
                lambda x: self.attack_to_state.get(x, 'Attack_Detected')
            )
            test_df['attack_category'] = test_df['attack_type'].map(
                lambda x: self.attack_categories.get(x, 'Unknown')
            )
            
            X_test = self.preprocess_features(test_df, fit=False)
            y_state_test = test_df['mdp_state'].values
            y_attack_test = test_df['attack_category'].values
            
            # State classifier accuracy
            state_pred = self.state_classifier.predict(X_test)
            state_acc = accuracy_score(y_state_test, state_pred)
            print(f"\nState Classifier Accuracy: {state_acc:.4f}")
            
            # Attack classifier accuracy
            attack_pred = self.attack_classifier.predict(X_test)
            attack_acc = accuracy_score(y_attack_test, attack_pred)
            print(f"Attack Type Classifier Accuracy: {attack_acc:.4f}")
            
            print("\nAttack Type Classification Report:")
            print(classification_report(y_attack_test, attack_pred))
        
        self.is_trained = True
        print("\nTraining complete!")
    
    def predict_from_features(self, features_df):
        """Predict state and attack type from features"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        X = self.preprocess_features(features_df, fit=False)
        
        state_pred = self.state_classifier.predict(X)
        attack_pred = self.attack_classifier.predict(X)
        state_proba = self.state_classifier.predict_proba(X)
        attack_proba = self.attack_classifier.predict_proba(X)
        
        return {
            'states': state_pred,
            'attack_types': attack_pred,
            'state_probabilities': state_proba,
            'attack_probabilities': attack_proba,
            'state_classes': self.state_classifier.classes_,
            'attack_classes': self.attack_classifier.classes_
        }
    
    def save_models(self, model_dir='data/models'):
        """Save trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.state_classifier, f'{model_dir}/state_classifier.pkl')
        joblib.dump(self.attack_classifier, f'{model_dir}/attack_classifier.pkl')
        joblib.dump(self.label_encoders, f'{model_dir}/label_encoders.pkl')
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        
        print(f"Models saved to {model_dir}/")
    
    def load_models(self, model_dir='data/models'):
        """Load trained models"""
        self.state_classifier = joblib.load(f'{model_dir}/state_classifier.pkl')
        self.attack_classifier = joblib.load(f'{model_dir}/attack_classifier.pkl')
        self.label_encoders = joblib.load(f'{model_dir}/label_encoders.pkl')
        self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
        self.is_trained = True
        
        print(f"Models loaded from {model_dir}/")
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features"""
        if not self.is_trained:
            return []
        
        importances = self.state_classifier.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        return [(self.feature_names[i], importances[i]) for i in indices]

if __name__ == '__main__':
    # Train models
    model = NSLKDDModel()
    model.train()
    model.save_models()
    
    # Show feature importance
    print("\nTop 10 Most Important Features:")
    for feature, importance in model.get_feature_importance(10):
        print(f"  {feature}: {importance:.4f}")
