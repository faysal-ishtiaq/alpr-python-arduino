import sqlite3

conn = sqlite3.connect('cars.db')
print("Opened database successfully")

conn.execute('''CREATE TABLE license_plates
       (ID INT PRIMARY KEY     NOT NULL,
       NUMBERS           TEXT    NOT NULL);''')

print("Table created successfully")




conn.execute("INSERT INTO license_plates (ID,NUMBERS) \
      VALUES (1, 'Paul')");
conn.execute("INSERT INTO license_plates (ID,NUMBERS) \
      VALUES (2, 'Paul')");
conn.execute("INSERT INTO license_plates (ID,NUMBERS) \
      VALUES (3, 'Paul')");
conn.execute("INSERT INTO license_plates (ID,NUMBERS) \
      VALUES (4, 'Paul')");


conn.commit()
print("Records created successfully")
conn.close()