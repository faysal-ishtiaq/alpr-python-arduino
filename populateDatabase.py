import sqlite3

conn = sqlite3.connect('cars.db')
print("Opened database successfully")

conn.execute('''CREATE TABLE license_plates
       (ID INT PRIMARY KEY     NOT NULL,
       NUMBERS           TEXT    NOT NULL);''')

print("Table created successfully")




conn.execute("INSERT INTO license_plates (ID,NUMBERS) \
      VALUES (1, '456CC')");
conn.execute("INSERT INTO license_plates (ID,NUMBERS) \
      VALUES (2, '999AA')");
conn.execute("INSERT INTO license_plates (ID,NUMBERS) \
      VALUES (3, '123BB')");


conn.commit()
print("Records created successfully")
conn.close()