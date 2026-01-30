# PostgreSQL Database Setup Guide

## Prerequisites

1. **Railway PostgreSQL Database**
   - Create a PostgreSQL database on Railway
   - Copy the `DATABASE_URL` connection string

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- Flask-SQLAlchemy (database ORM)
- Flask-Login (user authentication)
- Flask-Bcrypt (password hashing)
- psycopg2-binary (PostgreSQL adapter)
- python-dotenv (environment variables)
- Flask-Migrate (database migrations)

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your Railway PostgreSQL credentials:

```
DATABASE_URL=postgresql://user:password@host:port/dbname
SECRET_KEY=your-random-secret-key-here
FLASK_ENV=development
```

**Important:** Get the `DATABASE_URL` from your Railway PostgreSQL service.

### 3. Initialize the Database

Run the initialization script to create all tables:

```bash
python init_db.py
```

This will create:

- `users` table
- `analyses` table
- `predictions` table
- `user_statistics` table

### 4. Start the Application

```bash
python app.py
```

The application will run on `http://localhost:5000`

## First-Time Usage

1. **Register an Account**
   - Navigate to `/register`
   - Create your username, email, and password
   - Click "Create Account"

2. **Login**
   - Navigate to `/login`
   - Enter your credentials
   - You'll be redirected to the home page

3. **Upload Data**
   - Click "Batch Analysis" or navigate to `/upload`
   - Upload an NSL-KDD format file (`.txt` or `.csv`)
   - The system will analyze and save results to the database

4. **View Statistics**
   - Navigate to `/statistics` to see your personal analytics
   - All data is persistent and tied to your account

## Features

### User Authentication

- ✅ Secure password hashing with bcrypt
- ✅ Session management with Flask-Login
- ✅ "Remember me" functionality
- ✅ Protected routes (login required)

### Data Persistence

- ✅ All analyses saved to PostgreSQL
- ✅ Per-user statistics tracking
- ✅ Analysis history with detailed results
- ✅ Prediction records stored permanently

### User Features

- ✅ Personal profile page
- ✅ Analysis history
- ✅ View past analysis details
- ✅ Per-user statistics dashboard

## Database Schema

### Users Table

- `id` - Primary key
- `username` - Unique username
- `email` - Unique email
- `password_hash` - Hashed password
- `created_at` - Registration timestamp

### Analyses Table

- `id` - Primary key
- `user_id` - Foreign key to users
- `filename` - Uploaded file name
- `total_records` - Number of records analyzed
- `upload_timestamp` - When analysis was performed

### Predictions Table

- `id` - Primary key
- `analysis_id` - Foreign key to analyses
- `record_index` - Position in file
- `mdp_state` - Detected MDP state
- `attack_type` - Attack classification
- `recommended_action` - MDP recommendation
- `state_value` - MDP state value

### User Statistics Table

- `id` - Primary key
- `user_id` - Foreign key to users
- `total_analyzed` - Total records analyzed
- `attack_counts_json` - Attack type distribution (JSON)
- `state_counts_json` - State distribution (JSON)
- `action_counts_json` - Action distribution (JSON)
- `last_updated` - Last update timestamp

## Deployment to Railway

1. **Push to Git**

   ```bash
   git add .
   git commit -m "Add PostgreSQL database integration"
   git push
   ```

2. **Configure Railway**
   - Connect your GitHub repository
   - Add PostgreSQL service
   - Set environment variables:
     - `DATABASE_URL` (automatically set by Railway PostgreSQL)
     - `SECRET_KEY` (generate a random string)
     - `FLASK_ENV=production`

3. **Deploy**
   - Railway will automatically detect `requirements.txt`
   - It will run `python app.py` on deployment

4. **Initialize Database**
   - Run `python init_db.py` once after first deployment
   - You can do this via Railway's shell

## Troubleshooting

### Database Connection Error

- Verify `DATABASE_URL` is correct in `.env`
- Check Railway PostgreSQL service is running
- Ensure `psycopg2-binary` is installed

### Tables Not Created

- Run `python init_db.py` manually
- Check database permissions
- Verify Railway PostgreSQL credentials

### Login Not Working

- Clear browser cookies
- Check `SECRET_KEY` is set in `.env`
- Verify user exists in database

## Local Development (SQLite Fallback)

If you don't set `DATABASE_URL`, the app will use SQLite:

- Database file: `mdp_app.db`
- No PostgreSQL required
- Good for testing locally

## Migration from In-Memory to Database

**Note:** The old in-memory statistics are not migrated. All users start fresh after database integration.

## Security Notes

- Never commit `.env` file to Git (already in `.gitignore`)
- Use strong `SECRET_KEY` in production
- Passwords are hashed with bcrypt (never stored in plain text)
- SQL injection protection via SQLAlchemy ORM
