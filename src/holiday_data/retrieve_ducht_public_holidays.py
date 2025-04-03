from azureml.opendatasets import PublicHolidays
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd

# Define the date range
start_year = 2010
end_year = 2030

# Load public holidays dataset
holidays = PublicHolidays()

# Convert to pandas DataFrame
df = holidays.to_pandas_dataframe()

# Filter for Netherlands and the required year range
df = df[(df['countryOrRegion'] == 'Netherlands') &
        (df['date'] >= datetime(start_year, 1, 1)) &
        (df['date'] <= datetime(end_year, 12, 31))]

# Display the result
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/ovit2_gd_reporting"

# Create an engine
engine = create_engine(DATABASE_URL)

df.to_sql('holidays', engine, if_exists='replace', index=True)
