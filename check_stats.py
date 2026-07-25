import sqlite3
import os
from datetime import datetime, timedelta

DATABASE_PATH = 'text_formalizer.db'

def check_database_stats():
    """Check database statistics"""
    if not os.path.exists(DATABASE_PATH):
        print(f"❌ Database not found: {DATABASE_PATH}")
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    print("DATABASE STATISTICS")
    print("=" * 50)

    # Total conversions
    cursor.execute("SELECT COUNT(*) FROM conversions")
    total = cursor.fetchone()[0]
    print(f"\nTotal Conversions: {total}")

    # Recent conversions (last 24 hours)
    yesterday = datetime.now() - timedelta(days=1)
    cursor.execute("SELECT COUNT(*) FROM conversions WHERE timestamp >= ?", (yesterday.isoformat(),))
    recent = cursor.fetchone()[0]
    print(f"Recent (24h): {recent}")

    # Conversions by language
    cursor.execute("SELECT selected_language, COUNT(*) FROM conversions GROUP BY selected_language")
    languages = cursor.fetchall()
    print(f"\nBy Language:")
    for lang, count in languages:
        print(f"   - {lang}: {count}")

    # Conversions by formality level
    cursor.execute("SELECT formality_level, COUNT(*) FROM conversions WHERE formality_level IS NOT NULL GROUP BY formality_level")
    formality = cursor.fetchall()
    print(f"\nBy Formality Level:")
    for level, count in formality:
        print(f"   - {level}: {count}")
    
    # Average improvement score
    cursor.execute("SELECT AVG(improvement_score) FROM conversions WHERE improvement_score IS NOT NULL")
    avg_improvement = cursor.fetchone()[0]
    if avg_improvement:
        print(f"\nAverage Improvement: {avg_improvement:.1f}%")

    # Latest 5 conversions
    cursor.execute("""
        SELECT timestamp, selected_language, formality_level,
               SUBSTR(original_text, 1, 50) || '...' as preview
        FROM conversions
        ORDER BY timestamp DESC
        LIMIT 5
    """)
    recent_conversions = cursor.fetchall()
    print(f"\nLatest 5 Conversions:")
    for i, (ts, lang, formal, preview) in enumerate(recent_conversions, 1):
        print(f"   {i}. [{ts}] {lang} ({formal})")
        print(f"      {preview}")

    conn.close()
    print("\n" + "=" * 50)

if __name__ == '__main__':
    check_database_stats()
