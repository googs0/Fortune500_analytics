
pip install -U -q PyDrive
import numpy as np
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# f500 = raw data
# f500_all = cleaned data

# Define a function to load data from Google Drive with error handling
def load_data_from_drive(file_id, local_filename):
    try:
        # Authenticate and create the PyDrive client.
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)

        # Download the file from Google Drive
        file = drive.CreateFile({'id': file_id})
        file.GetContentFile(local_filename)
        print(f"File '{local_filename}' downloaded successfully.")
    except Exception as e:
        print(f"An error occurred while downloading the file: {str(e)}")

id_fortune500 = '1Nxs8P2MVMmOhDFwQve8rQZM2aLng_KmC'
f500_localname_csv = "fortune500.csv"
load_data_from_drive(id_fortune500, f500_localname_csv)

# Create dataframe
f500 = pd.read_csv("fortune500.csv")

f500_2021_drive = drive.CreateFile({'id':id_fortune500_2021})
f500_2021_localname_csv = "fortune500_2006_2021.csv"
f500_2021_drive.GetContentFile(f500_2021_localname_csv)

f500_drive = drive.CreateFile({'id':id_fortune500})
f500_localname_csv = "fortune500.csv"
f500_drive.GetContentFile(f500_localname_csv)

print(f"Fetching {f500_2021_localname_csv} and {f500_localname_csv}")

# Create dataframe
f500 = pd.read_csv("fortune500.csv")

# Get shape of dataset
f500.shape

# Use 0 index to get number of rows
print(f"Rows in dataset = {f500.shape[0]}")

# Head
f500.head()
# Tail
f500.tail()

# Datatypes of columns dataset
column_name.dtype

# f500 = f500.astype({"Profit (in millions)": float})
# f500 = f500.astype({"Revenue (in millions)": float})

f500.dtypes

f500[ f500["Profit (in millions)"] == "N.A." ]

# Clean NaN values
f500 = f500.replace("N.A.", np.NaN)
f500.fillna(0)
f500 = f500.astype({"Profit (in millions)": float})
f500.dtypes

pd.options.display.float_format = "{:,.2f}".format
f500.head()

# List Year and Company data
f500[ ["Year", "Company"]]

# Display tail of dataframe
ranked_492_sort = f500.tail(9)

# Get 0 index of the tail
ranked_492 = ranked_492_sort.iloc[0]

from pandas._libs.tslibs.offsets import YearBegin
# Search year == 1955 and get the 10th company index
year_1955 = f500[f500['Year'] == 1955].iloc[9]

# Clean data - rename columns
f500_rename = f500.rename({'Revenue (in millions)' : 'Revenue', 'Profit (in millions)' : 'Profit'}, axis='columns')

# validation check column names
f500_rename.head()

# Get by rank == 1
f500_number1 = f500[f500['Rank'] == 1]
print(f500_number1)

# Get by rank == 1 | rank == 2
f500_num1_num2 = f500[(f500['Rank'] == 1) | (f500['Rank'] == 2)]

f500_num1_num2.head(20)

# Get num 1 rank after 1999
num1_after1999 = f500[(f500['Rank'] == 1) & (f500['Year'] > 1999)]

# Create toprank_df dataframe
toprank_df = f500[f500['Rank'] == 1]

# head()
toprank_df.head()

# sum up all #1 companies
sum_f500 = toprank_df['Profit (in millions)'].sum()

# Get year 2005
f500_2005 = f500[f500['Year'] == 2005]

# Use max function to get highest profit
highest_profit_2005 = f500_2005['Profit (in millions)'].max()
highest_profit_2005_company = f500_2005['Company'].max()

highest_profit_2005

# Get data from 1980
f500_1980 = f500[f500['Year'] == 1980]

# Sum up total
f500_1980_revenue = f500_1980['Revenue (in millions)'].sum()

f500_1980_revenue

# Get groups of years
f500_all_years = f500[(f500['Year'] >= 1955) & (f500['Year'] <= 2005)]

# Add total revenue by groupby Revenue
f500_all_years.groupby('Revenue (in millions)').sum()

# Get group - year and profit with max function
max_profits_yearly = f500_all_years.groupby('Year')['Profit (in millions)'].max()

max_profits_yearly

# List all the #1 ranked Fortune 500 companies
all_num1 = f500[f500['Rank'] == 1]

all_num1

# List the #1 ranked companies AFTER year 2015
num1_after_2015 = f500[(f500['Rank'] == 1) & (f500['Year'] > 2015)]

num1_after_2015

# Get highest profit in dataframe (NO PROFIT COLUMN IN DF)
highest_profit = f500['Profit (in millions)'].max()

# Find index
highest_profit_index = f500['Profit (in millions)'].idxmax()

# Get company name
highest_profit_company = f500.iloc[highest_profit_index]['Company']

# Get Year
highest_profit_year = f500.iloc[highest_profit_index]['Year']


print(f"The highest profit company is {highest_profit_company} with a profit of {highest_profit} in {highest_profit_year}")

f500_all = f500_all.astype({"Revenue": float})
f500_all.dtypes
f500_all['Revenue'].str.isnumeric()
f500_all.dtypes

# Create a Seaborn plot
sns.scatterplot(data=f500, x='Year', y='Revenue', hue='Industry')

#Seaborn scatter plot to visualize the relationship between Year and Revenue
plt.figure(figsize=(12, 6))
sns.scatterplot(data=f500, x='Year', y='Revenue', hue='Industry')
plt.title('Fortune 500 Revenue Over Time')
plt.xlabel('Year')
plt.ylabel('Revenue (in millions)')
plt.show()

# Perform a linear regression analysis with statsmodels
X = f500['Year']
y = f500['Revenue']

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# Fit a linear regression model with statsmodels
model_stats = sm.OLS(y, X).fit()

# Print the summary of the regression analysis
print("Statsmodels Summary:")
print(model_stats.summary())

# Perform linear regression with scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)

# Make predictions with scikit-learn model
y_pred = model_sklearn.predict(X_test)

# Evaluate the scikit-learn model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nScikit-Learn Linear Regression Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Plot the data points and the regression line
plt.figure(figsize=(12, 6))
sns.scatterplot(data=f500, x='Year', y='Revenue', hue='Industry')
plt.title('Fortune 500 Revenue Over Time with Linear Regression')
plt.xlabel('Year')
plt.ylabel('Revenue (in millions)')
plt.plot(X['Year'], model_stats.predict(X), label='Statsmodels Linear Regression', color='red')
plt.plot(X_test['Year'], y_pred, label='Scikit-Learn Linear Regression', color='blue', linestyle='dashed')
plt.legend()
plt.show()