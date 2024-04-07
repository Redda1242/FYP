import sqlite3

def init_db(conn=None):
    # Allows passing a specific connection for testing
    own_connection = False
    if conn is None:
        conn = sqlite3.connect('feedback.db')
        own_connection = True
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                 image_id TEXT PRIMARY KEY,
                 classification_correct BOOLEAN,
                 deblur_satisfaction INTEGER,
                 perceived_quality TEXT,
                 additional_comments TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                 image_width INTEGER,
                 image_height INTEGER)''')
    conn.commit()
    if own_connection:
        conn.close()

def add_feedback_to_db(conn, image_id, classification_correct, deblur_satisfaction, additional_comments):
    c = conn.cursor()
    c.execute('''INSERT INTO feedback 
                 (image_id, classification_correct, deblur_satisfaction, additional_comments) 
                 VALUES (?, ?, ?, ?)''', 
                 (image_id, classification_correct, deblur_satisfaction, additional_comments))
    conn.commit()
