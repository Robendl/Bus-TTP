import json
import psycopg2

# Load JSON file
with open('schoolvakanties.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="ovit2_gd_reporting",
    user="postgres",
    password="postgres",
    host="localhost"
)
cur = conn.cursor()

# Insert data into school_vacations table
for item in data:
    for content in item['content']:
        school_year = content['schoolyear'].strip()

        for vacation in content['vacations']:
            vacation_name = vacation['type'].strip()

            for region in vacation['regions']:
                cur.execute("""
                    INSERT INTO school_vacations (region, year, start_date, end_date, vacation_name)
                    VALUES (%s, %s, %s, %s, %s);
                """, (region['region'], school_year, region['startdate'], region['enddate'], vacation_name))

# Commit and close
conn.commit()
cur.close()
conn.close()
