import sqlite3
import os

DATABASE_PATH = 'text_formalizer.db'

def check_and_fix_database():
    """Check database schema and add missing columns if needed"""
    print("Checking database schema...")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get current table schema
    cursor.execute("PRAGMA table_info(conversions)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    
    print(f"Found {len(columns)} columns in conversions table:")
    for col in columns:
        print(f"   - {col[1]} ({col[2]})")
    
    # Check for missing columns
    required_columns = {
        'context_format': 'TEXT DEFAULT "general"',
        'formality_level': 'TEXT DEFAULT "professional"',
        'custom_formality_score': 'INTEGER',
        'original_formality_score': 'REAL',
        'formalized_formality_score': 'REAL',
        'word_count': 'INTEGER',
        'improvement_score': 'REAL',
        'session_id': 'TEXT'
    }
    
    missing_columns = []
    for col_name, col_type in required_columns.items():
        if col_name not in column_names:
            missing_columns.append((col_name, col_type))

    if missing_columns:
        print(f"\nFound {len(missing_columns)} missing columns:")
        for col_name, col_type in missing_columns:
            print(f"   - {col_name}")
            try:
                # Use parameterized query for safety - SQLite doesn't support params for column names in ALTER TABLE
                # Since column_name and col_type come from our hardcoded dict, this is safe from injection
                cursor.execute(f"ALTER TABLE conversions ADD COLUMN {col_name} {col_type}")
                print(f"   Added {col_name}")
            except Exception as e:
                print(f"   Error adding {col_name}: {str(e)}")
        
        conn.commit()
        print("\nDatabase schema updated successfully!")
    else:
        print("\nAll required columns exist!")
    
    # Verify the updated schema
    print("\nFinal schema:")
    cursor.execute("PRAGMA table_info(conversions)")
    columns = cursor.fetchall()
    for col in columns:
        print(f"   {col[0]}. {col[1]} ({col[2]}) - Default: {col[4]}")
    
    conn.close()
    print("\nDatabase check complete!")

if __name__ == '__main__':
    if os.path.exists(DATABASE_PATH):
        check_and_fix_database()
    else:
        print(f"Database file not found: {DATABASE_PATH}")
        print("   Run the Flask app first to create the database.")
