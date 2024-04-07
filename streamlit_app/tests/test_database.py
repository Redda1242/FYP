import sqlite3
from database import init_db, add_feedback_to_db

def test_database_functions():
    # Connect to an in-memory database for testing
    connection = sqlite3.connect(':memory:')  

    # Initialize the database schema
    init_db(connection)

    # Test adding feedback
    add_feedback_to_db(connection, 'test_id', True, 4, 'Good')

    # Check if feedback was added correctly
    cursor = connection.cursor()
    cursor.execute("SELECT image_id, classification_correct, deblur_satisfaction, additional_comments FROM feedback WHERE image_id = 'test_id'")
    data = cursor.fetchone()

    # Asserts to ensure data is added correctly
    assert data is not None
    assert data[0] == 'test_id'  # Checking image_id
    assert data[1] == 1  # True should be stored as 1 in SQLite, check this
    assert data[2] == 4  # Checking satisfaction level
    assert data[3] == 'Good'  # Checking additional comments

    # Close the connection
    connection.close()
