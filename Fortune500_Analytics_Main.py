import numpy as np
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Authenticate using your Google account
gauth = GoogleAuth()

# Create and authorize a local webserver for OAuth2.0
gauth.LocalWebserverAuth()

# Create a GoogleDrive instance with authenticated GoogleAuth instance
drive = GoogleDrive(gauth)

# Authenticate Google Drive
def authenticate_google_drive():
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive

# Download a Fortune 500 Data file
def download_file(drive, file_id, local_filename):
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile(local_filename)

# Authenticate Google Drive and download files
drive = authenticate_google_drive()
id_fortune500 = '1Nxs8P2MVMmOhDFwQve8rQZM2aLng_KmC'
id_fortune500_2021 = '1rm_p7Kq3vSxbguTNiIpRX7AIPrCrDq-Z'
download_file(drive, id_fortune500, "fortune500.csv")
download_file(drive, id_fortune500_2021, "fortune500_2006_2021.csv")

# Read data into a DataFrame
f500 = pd.read_csv("fortune500.csv")

# Data cleaning
f500 = f500.replace("N.A.", np.NaN).fillna(0)
f500["Profit (in millions)"] = f500["Profit (in millions)"].astype(float)
f500["Revenue (in millions)"] = f500["Revenue (in millions)"].astype(float)

# Set display format for float values
pd.options.display.float_format = "{:,.2f}".format

# -- Metrics and Data Analysis --
# Number of rows in the dataset
num_rows = f500.shape[0]
print(f"Number of rows in the dataset: {num_rows}")

# Top 10 rows
print(f"Top 10 rows of the dataset:\n {f500.head(10)}")

# Bottom 10 rows
print(f"Bottom 10 rows of the dataset:\n{f500.tail(10)}")

# Data types of columns
print(f"Data types of columns in the dataset:\n{f500.dtypes}")

# Total profit of all #1 ranked companies
total_profit_num1 = f500[f500['Rank'] == 1]['Profit (in millions)'].sum()
print(f"Total profit of all #1 ranked companies: ${total_profit_num1:.2f} million\n")

# Highest profit company and year
highest_profit = f500['Profit (in millions)'].max()
highest_profit_index = f500['Profit (in millions)'].idxmax()
highest_profit_company = f500.loc[highest_profit_index, 'Company']
highest_profit_year = f500.loc[highest_profit_index, 'Year']
print(f"""The company with the highest profit is {highest_profit_company} with 
a profit of ${highest_profit:.2f} million in {highest_profit_year}""")

# Total revenue in the year 1980
total_revenue_1980 = f500[f500['Year'] == 1980]['Revenue (in millions)'].sum()
print(f"Total revenue in the year 1980: ${total_revenue_1980:.2f} million\n")

# Maximum profit by year
max_profits_yearly = f500.groupby('Year')['Profit (in millions)'].max()
print(f"Maximum profit by year: {max_profits_yearly}\n")

# List of all #1 ranked Fortune 500 companies
all_num1 = f500[f500['Rank'] == 1][['Year', 'Company']]
print(f"List of all #1 ranked Fortune 500 companies: {all_num1}")

# List of #1 ranked companies after the year 2015
num1_after_2015 = f500[(f500['Rank'] == 1) & (f500['Year'] > 2015)][['Year', 'Company']]
print(f"List of #1 ranked companies after the year 2015: {num1_after_2015}\n")

# Total revenue by year
total_revenue_by_year = f500.groupby('Year')['Revenue (in millions)'].sum()
print(f"Total revenue by year: {total_revenue_by_year}\n")