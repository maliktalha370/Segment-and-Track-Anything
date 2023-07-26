import sqlite3

# Connect to the database
conn = sqlite3.connect('SegTracking.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Execute a SELECT query to retrieve the data from the table
cursor.execute('SELECT video_name, metadata FROM videos')

# Fetch all the rows returned by the query
rows = cursor.fetchall()

# Iterate over the rows and print the values
for row in rows:
    name = row[0]
    metadata = row[1]
    print(f"Name: {name}, Metadata: {metadata}")

# Close the database connection
conn.close()
