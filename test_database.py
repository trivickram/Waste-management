"""
Test script to verify database functionality
"""
import sqlite3
import os

DB_PATH = 'waste_monitoring.db'

def check_database():
    """Check if database exists and show contents"""
    if not os.path.exists(DB_PATH):
        print("❌ Database does not exist yet. Run the app first to create it.")
        return
    
    print("✅ Database found!")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute("SELECT COUNT(*) FROM waste_records")
    count = cursor.fetchone()[0]
    print(f"📊 Total records: {count}")
    
    if count > 0:
        # Show last 5 records
        cursor.execute("SELECT id, timestamp, classification, confidence FROM waste_records ORDER BY id DESC LIMIT 5")
        records = cursor.fetchall()
        
        print("\n🗂️ Last 5 records:")
        print("-" * 80)
        for record in records:
            print(f"ID: {record[0]} | Time: {record[1]} | Type: {record[2]} | Confidence: {record[3]:.2f}%")
        print("-" * 80)
    
    conn.close()

if __name__ == "__main__":
    check_database()
