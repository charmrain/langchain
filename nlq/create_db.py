import sqlite3

# Connect to an existing database or create a new one (in this case, "mydatabase.db"):
conn = sqlite3.connect("mydatabase.db")

#  Connect to an existing database or create a new one (in this case, "mydatabase.db"):
conn = sqlite3.connect("mydatabase.db")

cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    age INTEGER,
                    department TEXT
                )''')


conn.commit()
conn.close()
