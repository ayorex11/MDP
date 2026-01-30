"""
Database initialization script
Run this to create all database tables
"""

def init_database():
    """Initialize the database"""
    # Import here to avoid circular imports
    from app import app
    from models import db
    
    with app.app_context():
        print("Creating database tables...")
        db.create_all()
        print("✓ Database tables created successfully!")
        
        # Check if tables were created
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"\nCreated tables: {', '.join(tables)}")

if __name__ == '__main__':
    init_database()

