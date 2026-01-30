# MDP Cyber-Attack Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Intelligent cyber-attack prediction and defense recommendation system using Markov Decision Processes (MDPs) and Machine Learning.

🔗 **Live Demo:** [cybermarkov.xyz](https://cybermarkov.xyz)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This system combines **Markov Decision Processes** with **Machine Learning** to predict cyber-attacks and recommend optimal defensive actions. Built for academic research on AI-driven cybersecurity, it demonstrates how sequential decision-making under uncertainty can enhance network defense strategies.

### Key Capabilities:

- **MDP-Based Decision Making:** Uses Value Iteration algorithm to compute optimal defense policies
- **Machine Learning Integration:** Random Forest classifiers trained on NSL-KDD dataset
- **Real-Time Predictions:** Instant attack state classification and action recommendations
- **Hybrid Approach:** Combines rule-based and data-driven methods for robust performance
- **Multi-User Support:** User authentication, analysis history, and personalized statistics
- **Interactive Visualizations:** Charts and graphs for prediction probabilities and results

---

## ✨ Features

### Core Functionality

- ✅ **Value Iteration Algorithm:** Computes optimal policies for 6-state, 5-action MDP
- ✅ **State Classification:** ML-powered mapping from network features to security states
- ✅ **Attack Type Detection:** Identifies specific attack categories (DoS, Probe, R2L, U2R)
- ✅ **Risk Assessment:** Automatic risk level calculation (LOW, MEDIUM, HIGH)
- ✅ **Action Recommendations:** Suggests optimal defensive measures with value justification

### User Features

- 🔐 **Secure Authentication:** Password hashing, session management, remember me
- 📊 **Analysis Dashboard:** Track prediction history and aggregate statistics
- 📁 **File Upload:** Batch analysis of NSL-KDD format datasets
- 📈 **Visualizations:** Real-time charts for probabilities and distributions
- 📝 **Export Results:** Download predictions and analysis reports

### Technical Features

- ⚡ **Pre-Trained Models:** MDP trains at startup for instant predictions
- 🗄️ **Database Persistence:** SQLAlchemy ORM with SQLite/PostgreSQL support
- 🎨 **Responsive UI:** Mobile-friendly interface with modern design
- 🚀 **Cloud-Ready:** Railway.app deployment configuration included
- 📦 **Modular Architecture:** Clean separation of concerns for maintainability

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────┐
│           PRESENTATION LAYER                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│   │   Home   │  │ Predict  │  │   Results    │ │
│   └──────────┘  └──────────┘  └──────────────┘ │
└──────────────────────┬──────────────────────────┘
                       │ HTTP
                       ↓
┌─────────────────────────────────────────────────┐
│          APPLICATION LAYER (Flask)               │
│  ┌───────────────┐      ┌────────────────────┐ │
│  │  MDP Solver   │      │   ML Classifier    │ │
│  │ Value Iteration│  ←→  │  Random Forest     │ │
│  └───────────────┘      └────────────────────┘ │
│                                                  │
│  ┌───────────────────────────────────────────┐ │
│  │     Authentication & Session Manager      │ │
│  └───────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────┘
                       │ SQLAlchemy ORM
                       ↓
┌─────────────────────────────────────────────────┐
│          DATA LAYER                              │
│  ┌──────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Users │  │ Analyses │  │   Predictions    │ │
│  └──────┘  └──────────┘  └──────────────────┘ │
│           NSL-KDD Dataset + Trained Models      │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/mdp-cyber-attack.git
cd mdp-cyber-attack
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Initialize Database

```bash
python init_db.py
```

This creates the SQLite database and necessary tables.

### Step 5: Train ML Models (Optional - Pre-trained models included)

```bash
python ml_model.py
```

This trains the Random Forest classifiers on NSL-KDD data (~5 minutes).

### Step 6: Run Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

---

## 📖 Usage

### Quick Start

1. **Register Account**
   - Navigate to `/register`
   - Create username and password (8+ chars, mixed case, numbers, special chars)

2. **Manual Prediction**
   - Go to `/predict`
   - Enter network traffic level (0-10,000 req/s)
   - Enter suspicious indicators count (0-50)
   - Click "Analyze & Predict"

3. **View Results**
   - See current security state
   - Review recommended defensive action
   - Check risk level (LOW/MEDIUM/HIGH)
   - Examine probability distributions

4. **Batch Analysis (Optional)**
   - Go to `/upload`
   - Upload NSL-KDD format file
   - System analyzes all records
   - View aggregate statistics

### Example Usage

**Scenario 1: Normal Traffic**
```
Traffic Level: 500 req/s
Suspicious Indicators: 0
→ State: Normal
→ Action: No_Action
→ Risk: LOW
```

**Scenario 2: DDoS Attack**
```
Traffic Level: 8000 req/s
Suspicious Indicators: 15
→ State: Attack_Detected
→ Action: Block_IPs
→ Risk: HIGH
```

---

## 📊 Dataset

### NSL-KDD Intrusion Detection Dataset

The system uses the NSL-KDD dataset, a refined version of KDD Cup 1999 data.

**Location:** `data/KDDTrain+.TXT` and `data/KDDTest+.TXT`

**Statistics:**
- Training: 125,973 records
- Testing: 22,544 records
- Features: 41 (numerical + categorical)
- Attack Types: 39 specific attacks in 4 categories
  - **DoS:** neptune, smurf, pod, teardrop, land, back, etc.
  - **Probe:** portsweep, ipsweep, nmap, satan, saint, mscan
  - **R2L:** guess_passwd, ftp_write, imap, phf, warezmaster, etc.
  - **U2R:** buffer_overflow, loadmodule, rootkit, perl, etc.

**Download:**
```bash
# Dataset is included in repository
# If missing, download from:
# https://www.unb.ca/cic/datasets/nsl.html
```

**Feature Categories:**
1. Basic: duration, protocol_type, service, flag, src_bytes, dst_bytes
2. Content: hot, num_failed_logins, logged_in, root_shell, etc.
3. Time-based: count, srv_count, error rates
4. Host-based: dst_host counts and rates

---

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** - Programming language
- **Flask 3.0** - Web framework
- **SQLAlchemy** - ORM for database
- **Flask-Login** - User authentication

### Machine Learning
- **scikit-learn 1.3.0** - Random Forest classifiers
- **NumPy 1.24.3** - Numerical computing
- **pandas 2.0.3** - Data manipulation
- **joblib** - Model serialization

### Visualization
- **Matplotlib 3.7.2** - Chart generation
- **Seaborn** - Statistical plots

### Database
- **SQLite** - Development database
- **PostgreSQL** - Production (Railway deployment)

### Security
- **Werkzeug** - Password hashing (PBKDF2)
- **Flask session management** - Secure cookies

### Deployment
- **Railway.app** - Cloud hosting platform
- **Gunicorn** - WSGI HTTP server

---

## 📁 Project Structure

```
MDP/
├── app.py                      # Main Flask application
├── ml_model.py                 # ML model training & inference
├── models.py                   # Database models (SQLAlchemy)
├── config.py                   # Configuration settings
├── init_db.py                  # Database initialization
├── requirements.txt            # Python dependencies
├── railway.toml               # Railway deployment config
├── .env                       # Environment variables (not in git)
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── DATABASE_SETUP.md          # Database documentation
│
├── data/                      # Datasets and models
│   ├── KDDTrain+.TXT         # Training dataset
│   ├── KDDTest+.TXT          # Test dataset
│   ├── feature_names.txt     # Feature definitions
│   ├── models/               # Trained ML models
│   │   ├── state_classifier.pkl
│   │   ├── attack_classifier.pkl
│   │   ├── label_encoders.pkl
│   │   └── scaler.pkl
│   └── uploads/              # User uploaded files
│
├── static/                    # Static assets
│   └── style.css             # Stylesheet
│
├── templates/                 # HTML templates
│   ├── index.html            # Home page
│   ├── login.html            # Login page
│   ├── register.html         # Registration
│   ├── predict.html          # Prediction form
│   ├── results.html          # Results display
│   ├── upload.html           # File upload
│   ├── analyze.html          # Analysis results
│   ├── profile.html          # User profile
│   ├── history.html          # Analysis history
│   ├── view_analysis.html    # Single analysis view
│   └── statistics.html       # Statistics dashboard
│
└── venv/                      # Virtual environment (not in git)
```

---

## 🔌 API Documentation

### Authentication Endpoints

**Register User**
```
POST /register
Content-Type: application/x-www-form-urlencoded

username=testuser&email=test@example.com&password=SecurePass123!&confirm_password=SecurePass123!
```

**Login**
```
POST /login
Content-Type: application/x-www-form-urlencoded

username=testuser&password=SecurePass123!&remember=yes
```

**Logout**
```
GET /logout
```

### Prediction Endpoints

**Manual Prediction**
```
POST /results
Content-Type: application/x-www-form-urlencoded
Authorization: Session cookie required

traffic=8000&suspicious_indicators=15

Response: HTML page with prediction results
```

**File Analysis**
```
POST /analyze
Content-Type: multipart/form-data
Authorization: Session cookie required

file=<NSL-KDD format file>

Response: HTML page with batch analysis results
```

### User Data Endpoints

**View Profile**
```
GET /profile
Authorization: Session cookie required

Response: User statistics and analysis count
```

**Analysis History**
```
GET /history
Authorization: Session cookie required

Response: List of all user analyses
```

**View Specific Analysis**
```
GET /analysis/<analysis_id>
Authorization: Session cookie required

Response: Detailed predictions for specific analysis
```

**Statistics Dashboard**
```
GET /statistics
Authorization: Session cookie required

Response: Aggregate charts and metrics
```

---

## 🌐 Deployment

### Local Development

```bash
# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Run with auto-reload
flask run --debug
```

### Railway.app Deployment

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/mdp-cyber-attack.git
   git push -u origin main
   ```

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Python app

3. **Add Environment Variables**
   ```
   SECRET_KEY=your-secret-key-here
   DATABASE_URL=postgresql://... (Railway provides this)
   FLASK_ENV=production
   ```

4. **Add Custom Domain**
   - In Railway dashboard: Settings → Domains
   - Add custom domain: cybermarkov.xyz
   - Configure DNS at your registrar:
     - CNAME record: www → [railway-url]
     - A record: @ → Railway IP

5. **Initialize Database**
   ```bash
   railway run python init_db.py
   railway run python ml_model.py
   ```

### Environment Variables

Create `.env` file (not committed to git):

```env
SECRET_KEY=your-super-secret-key-change-this-in-production
DATABASE_URL=sqlite:///app.db
FLASK_ENV=development
UPLOAD_FOLDER=data/uploads
MAX_CONTENT_LENGTH=16777216
ALLOWED_EXTENSIONS=txt
```

---

## 🧪 Testing

### Run Unit Tests

```bash
python -m pytest tests/
```

### Manual Testing Scenarios

**Test 1: Normal Traffic**
- Traffic: 500, Indicators: 0
- Expected: Normal state, No_Action

**Test 2: Suspicious Traffic**
- Traffic: 3000, Indicators: 7
- Expected: High_Suspicious, Rate_Limit or Block_IPs

**Test 3: Confirmed Attack**
- Traffic: 9000, Indicators: 20
- Expected: Attack_Detected, Block_IPs

**Test 4: File Upload**
- Upload: KDDTest+.TXT (first 100 lines)
- Expected: Classification results, charts generated

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**ADEWALE ADEWALE AYOMIDE** - *Project Lead* - [GitHub](https://github.com/ayorex11)

**Team Members:**
- AROYEWUN OLUWATOBILOBA JOHN 
- OLORUNFEMI EMMANUEL
- ADEBUMOYE ISAAC ADEAYO
- OSHAI UCHECHUKWU
- NWAMU PEACE
- OJEIKERE ALEXANDER 
- OHAI DANIELLA EBUBECHUKWU 
- SUCCESS GABRIEL AYOMIDE 
- IFEBAJO OLUWATENIOLA PHILIP
- UMEMBA WILSON 
- OLAOLUWOYE OLAOLUWAKIITAN
- OGUNGBE FRENZY MADESOLA
- NYECHE DESTINY
- IYERE DAVID OSEZUA
- TOPSON JOHN CHIBUIKE
- ADEGOKE GBENGA SAMUEL
- OBI CHOSEN
- EGBAUNU PROMISE EMMANUEL 
- UBOSIUKWA NONSO VICTOR 
- AFOLABI FOLUWASO MORAYO


---

## 🙏 Acknowledgments

- **NSL-KDD Dataset:** Canadian Institute for Cybersecurity, University of New Brunswick
- **Flask Framework:** Pallets Projects
- **scikit-learn:** scikit-learn developers
- **Railway.app:** For hosting platform
- **Dr. Omobolaji Odegbesan:** Course instructor and advisor
- **Cybersecurity Department:** For project support

---

## 📚 References

**Academic Papers:**
- Tavallaee, M., et al. (2009). "A detailed analysis of the KDD CUP 99 data set." IEEE Symposium on Computational Intelligence for Security and Defense Applications.
- Bellman, R. (1957). "A Markovian Decision Process." Journal of Mathematics and Mechanics.
- Puterman, M. L. (2014). "Markov Decision Processes: Discrete Stochastic Dynamic Programming." John Wiley & Sons.

**Datasets:**
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html

**Documentation:**
- Flask: https://flask.palletsprojects.com/
- scikit-learn: https://scikit-learn.org/
- SQLAlchemy: https://www.sqlalchemy.org/

---

## 📞 Contact

For questions or support:
- Email: adewalemaxwell11@gmail.com
- Project Repository: https://github.com/ayorex11/MDP
- Live Demo: https://cybermarkov.xyz

---

## 🎓 Academic Context

This project was developed as the term paper for the Artificial Intelligence course (Academic Session 2025/2026) under the supervision of Dr. Omobolaji Odegbesan, Department of Computer Science/Cybersecurity.

**Course Learning Objectives Met:**
- ✅ Deep theoretical understanding of MDPs
- ✅ Practical AI system implementation
- ✅ Research depth and academic rigor
- ✅ Ethical considerations in AI
- ✅ Real-world application demonstration

---

**⭐ Star this repository if you found it helpful!**

**🔗 Live Demo:** [cybermarkov.xyz](https://cybermarkov.xyz)