import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import seasonal_decompose

logging.basicConfig(level=logging.DEBUG)


def download_and_load_data(url, local_filename):
    try:
        # Download CSV from URL and save to local file
        response = requests.get(url)
        response.raise_for_status()
        with open(local_filename, 'wb') as file:
            file.write(response.content)
            print(f"File '{local_filename}' downloaded successfully.")

        # Read CSV into a DataFrame
        data = pd.read_csv(local_filename)
        return data

    except requests.exceptions.RequestException as req_err:
        logging.error(f"An unexpected error occurred during download: {req_err}")
        return pd.DataFrame()
    except FileNotFoundError:
        logging.error(f"File not found: {local_filename}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logging.error("The dataset is empty. Please check the data source.")
        return pd.DataFrame()
    except pd.errors.ParserError as pe:
        logging.error(f"Error parsing the CSV file: {pe}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()


# URL and local filenames for Fortune 500 data
raw_url_1 = 'https://github.com/googs0/Fortune500LinearExplorer/raw/main/fortune500.csv'
local_filename_1 = 'fortune500.csv'

raw_url_2 = 'https://github.com/googs0/Fortune500LinearExplorer/raw/main/fortune500-2005-2021.csv'
local_filename_2 = 'fortune500-2005-2021.csv'

f500_1 = download_and_load_data(raw_url_1, local_filename_1)
f500_2 = download_and_load_data(raw_url_2, local_filename_2)

# Shape of unmerged datasets
print(f"Shape of f500_1 before concatenation: {f500_1.shape}")
print(f"Shape of f500_2 before concatenation: {f500_2.shape}")

# Concatenate datasets
f500_combined = pd.concat([f500_1, f500_2], ignore_index=True)
pd.options.display.float_format = "{:,.2f}".format
print("Shape of f500_combined after concatenation:", f500_combined.shape)


def clean_data(data):
    try:
        # Rename columns
        data.rename(
            columns={'Profit (in millions)': 'Profit', 'Revenue (in millions)': 'Original_Revenue'}, inplace=True
        )

        # Identify and drop duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]

        # Shape before cleaning
        print(f"Shape before cleaning: {data.shape}")
        print(data.head())

        # Clean 'Profit' column
        data['Profit'] = pd.to_numeric(
            data['Profit'].replace('[\\$,]', '', regex=True)
            .replace('N.A.', np.nan),
            errors='coerce'
        )

        # Clean 'Revenue'
        data['Revenue'] = pd.to_numeric(data['Original_Revenue'].replace(',', '', regex=True), errors='coerce')

        # Drop duplicates and handle NaN values
        cleaned_data = data.drop_duplicates().dropna(subset=['Year', 'Revenue', 'Profit'])

        # After cleaning quickview metrics
        print("Shape after cleaning:", cleaned_data.shape)
        print("Cleaned DataFrame:")
        print(cleaned_data)

        return cleaned_data

    except pd.errors.EmptyDataError:
        logging.error("The dataset is empty. Please check the data source.")
        return pd.DataFrame()
    except pd.errors.ParserError as parser_err:
        logging.error(f"Error parsing the CSV file: {parser_err}")
        return pd.DataFrame()
    except Exception as general_exception:
        logging.error(f"An unexpected error occurred: {general_exception}")
        return pd.DataFrame()


f500_clean = clean_data(f500_combined)
print("Shape of f500_clean data_clean function:", f500_clean.shape)


def validate_data(data):
    # Domain validation
    if 'Year' in data.columns:
        min_year = 1955
        max_year = 2021
        invalid_years = data[(data['Year'] < min_year) | (data['Year'] > max_year)]['Year'].unique()
        if len(invalid_years) > 0:
            logging.warning(
                f"Warning: Invalid years found in 'Year' column: {invalid_years}"
                f"Expected range: {min_year} - {max_year}"
            )

    # Datatypes validation
    expected_dtypes = {'Year': int, 'Rank': int, 'Company': object, 'Profit': float, 'Revenue': float}
    for column, expected in expected_dtypes.items():
        if column in data.columns and data[column].dtype != expected:
            logging.warning(f"Warning: Data type of column: {column} is not as expected: {expected_dtypes}")


validate_data(f500_clean)


def handle_outliers(data, column, threshold=3):
    """
    Detect and handle outliers in specified column using Z-scores.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to handle outliers.
        threshold (float): Z-score threshold for outlier detection.

    Returns:
        pd.DataFrame: DataFrame with outliers handled based on the specified threshold.
        pd.DataFrame: DataFrame containing the detected outliers.
    """

    # Calculate Z-scores for the specified column
    data[f'{column}_ZScore'] = zscore(data[column])

    # Identify and store outliers
    outliers = data[data[f'{column}_ZScore'].abs() > threshold]

    # Remove outliers from the data using .loc
    data = data.loc[data[f'{column}_ZScore'].abs() <= threshold].copy()

    return data, outliers


def correlation_matrix(data):
    plt.figure(figsize=(10, 8))
    numeric_data = data.select_dtypes(include=['number'])
    correlation_mtx = numeric_data.corr()
    sns.heatmap(correlation_mtx, annot=True, cmap="coolwarm_r", linewidths=.5, fmt=".2f")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Correlation Matrix", fontsize=16)
    plt.axhline(y=0, color='k', linewidth=1)
    plt.axvline(x=0, color='k', linewidth=1)

    plt.show()
    return True


def time_series_analysis(data, decomposition_model='additive', period=2):
    plt.figure(figsize=(12, 10))
    custom_palette = sns.color_palette("viridis", n_colors=4)

    result = seasonal_decompose(data['Profit'], model=decomposition_model, period=period)

    # Raw data
    plt.subplot(5, 1, 1)
    sns.lineplot(x=data['Year'], y=data['Profit'], color=custom_palette[0])
    plt.title('Raw Data')

    # Observed
    plt.subplot(5, 1, 2)
    sns.lineplot(x=result.observed.index, y=result.observed, color=custom_palette[1])
    plt.title('Observed')

    # Trend
    plt.subplot(5, 1, 3)
    sns.lineplot(x=result.trend.index, y=result.trend, color=custom_palette[2])
    plt.title('Trend')

    # Residual
    plt.subplot(5, 1, 5)
    sns.lineplot(x=result.resid.index, y=result.resid, color='orange')
    plt.title('Residual')

    # Seasonal (manual compute)
    seasonal_component = data['Profit'] - result.trend - result.resid
    plt.subplot(5, 1, 4)
    sns.lineplot(x=data['Year'], y=seasonal_component, color=custom_palette[3])
    plt.title('Seasonal')
    plt.tight_layout()

    plt.show()
    return True


def pair_plot(data):
    pair_palette = sns.color_palette("coolwarm")
    plot_kws = {'color': pair_palette[0]}
    sns.pairplot(data, plot_kws=plot_kws)

    plt.show()
    return True


def box_plot(data, sample_size=100):
    try:
        if sample_size > len(data):
            logging.error("Sample size is greater than number of rows")
            sample_size = len(data)

            plt.figure(figsize=(14, 10))
            plt.rcParams.update({'figure.autolayout': True})

            sns.set(style="whitegrid", font_scale=1.2)

            ax = sns.boxplot(
                x='Revenue',
                y='Profit',
                data=data.sample(n=sample_size),
                fill=False,
                width=1,
                linewidth=3,
                hue='Profit',
                dodge=False,
                fliersize=8,
                whis=0.8,
            )

            ax.yaxis.grid(True)

            ax.set_xlabel('Revenue', fontsize=15)
            ax.set_ylabel('Profit', fontsize=15)
            ax.set_title('Boxplot of Profit by Revenue', fontsize=18)
            ax.legend().set_visible(False)

            plt.show()
            return True

    except Exception as e:
        logging.error(f'Error in box_plot: {e}')
        return False


def cluster_analysis(data):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=3, n_init=10)
    data['Cluster'] = kmeans.fit_predict(data[['Profit', 'Revenue']])

    custom_palette = sns.color_palette("tab10", 3)
    plt.figure(figsize=(12, 8))

    # Scatter plot (hue to visualize clusters)
    sns.scatterplot(x='Profit', y='Revenue', hue='Cluster', data=data, palette=custom_palette, s=100, alpha=0.8)
    plt.title("Cluster Analysis", fontsize=16)
    plt.xlabel("Profit", fontsize=14)
    plt.ylabel("Revenue", fontsize=14)

    legend = plt.legend(title='Cluster', loc='upper left')
    legend.get_title().set_fontsize('14')

    plt.show()
    return True


def plot_scatter(data):
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("viridis", as_cmap=True)

    sns.scatterplot(data=data, x='Year', y='Revenue', hue='Profit', palette=palette,
                    size='Profit', sizes=(50, 200), alpha=0.7, s=100)

    plt.title('Fortune 500 Revenue and Profit Over the Years', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Revenue', fontsize=14)

    legend = plt.legend(title='Profit', loc='upper left')
    legend.get_title().set_fontsize('14')

    plt.show()
    return True


def data_analysis(data):
    # Validation check column names
    print("Validation check column names:")
    print(data.head())
    print("Shape of the cleaned dataset:", data.shape)
    print(f"Data from {f500_clean['Year'].min()} to {f500_clean['Year'].max()}")
    print(f500_clean.columns)
    print(f"Number of rows in the cleaned dataset: {f500_clean.shape[0]}")
    print(f"Number of NaN values in 'Revenue': {f500_clean['Revenue'].isna().sum()}")

    # Dataset info
    print(f"Rows in the dataset: {data.shape[0]}")
    print(f"Datatypes of columns in the dataset:{data.dtypes}")

    # Outlier Detection and Handling
    data, outliers = handle_outliers(data, 'Profit')

    # Unique values in the 'Year' column
    print("Unique values in the 'Year' column:", data['Year'].unique())

    # Company in 10th row where 'Year' is 1955
    year_1955 = data[data['Year'] == 1955].head(10)
    print("Companies in the 10th row where 'Year' is 1955:")
    print(year_1955)

    # Number 1 ranked company
    f500_number1 = data[data['Rank'] == 1]
    print("Number 1 ranked company:")
    print(f500_number1)

    # Number 1 and 2 ranked companies
    f500_num1_num2 = data[(data['Rank'] == 1) | (data['Rank'] == 2)].head(20)
    print("Number 1 and 2 ranked companies:")
    print(f500_num1_num2)

    # Number 1 rank after 1999
    num1_after1999 = data[(data['Rank'] == 1) & (data['Year'] > 1999)]
    print("Number 1 rank after 1999:")
    print(num1_after1999)

    # All top-ranked Fortune 500 companies
    all_num1 = data[data['Rank'] == 1]
    print("All Top Ranked #1 Companies:")
    print(all_num1)

    # All top-ranked companies after year 2015
    num1_after_2015 = data[(data['Rank'] == 1) & (data['Year'] > 2015)]
    print("List of #1 ranked companies after 2015:")
    print(num1_after_2015)

    # Highest profit in the dataframe
    highest_profit = data['Profit'].max()
    highest_profit_index = data['Profit'].idxmax()
    highest_profit_company = data.loc[highest_profit_index, 'Company']
    highest_profit_year = data.loc[highest_profit_index, 'Year']

    print(f"""The highest profit company is {highest_profit_company} 
    with a profit of {highest_profit} in {highest_profit_year}""")

    # Outlier Detection
    data['Profit_ZScore'] = zscore(data['Profit'])

    # Plots
    plot_scatter(data)
    correlation_matrix(data)
    cluster_analysis(data)
    time_series_analysis(data, decomposition_model='additive', period=2)
    pair_plot(data)
    regression_analysis(data, x_columns=['Year', 'Revenue'], y_column='Profit', degree=2)
    box_plot(data)

    return data


def regression_analysis(data, x_columns, y_column, degree=3, include_outliers=False, threshold=3):
    try:
        # Extract features (x) and target variable (y)
        x = data[x_columns]
        y = data[y_column]

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Linear Regression
        linear_model = LinearRegression()
        linear_model.fit(x_train, y_train)
        y_prediction_linear = linear_model.predict(x_test)

        # Polynomial Regression
        poly_degree = degree  # Adjust the degree as needed
        poly = PolynomialFeatures(degree=poly_degree)
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.transform(x_test)

        poly_model = LinearRegression()
        poly_model.fit(x_train_poly, y_train)
        y_prediction_poly = poly_model.predict(x_test_poly)

        # Evaluate Linear
        mse_linear = mean_squared_error(y_test, y_prediction_linear)
        r2_linear = r2_score(y_test, y_prediction_linear)
        print("Linear Regression Results:")
        print(f"Mean Squared Error: {mse_linear:.2f}")
        print(f"R-squared: {r2_linear:.2f}")

        # Evaluate Polynomial
        mse_poly = mean_squared_error(y_test, y_prediction_poly)
        r2_poly = r2_score(y_test, y_prediction_poly)
        print("\nPolynomial Regression Results:")
        print(f"Mean Squared Error: {mse_poly:.2f}")
        print(f"R-squared: {r2_poly:.2f}")

        plt.figure(figsize=(15, 6))

        # Linear Regression
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=x_test[x_columns[0]], y=y_test, label='True Values')
        sns.lineplot(x=x_test[x_columns[0]], y=y_prediction_linear, color='red', label='Linear Regression')
        plt.title('Linear Regression')
        plt.xlabel(x_columns[0])
        plt.ylabel(y_column)
        plt.legend()

        # Polynomial Regression
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=x_test[x_columns[0]], y=y_test, label='True Values')
        sns.lineplot(x=x_test[x_columns[0]], y=y_prediction_poly, color='green',
                     label=f'Polynomial Regression (Degree {poly_degree})')
        plt.title('Polynomial Regression')
        plt.xlabel(x_columns[0])
        plt.ylabel(y_column)
        plt.legend()
        plt.tight_layout()

        plt.show()

        if include_outliers:
            # Calculate Z-scores for Profit
            data['Profit_ZScore'] = zscore(data['Profit'])

            # Outlier Detection
            outliers = data[data['Profit_ZScore'].abs() > threshold]
            success_flag = True
            return data, outliers, success_flag
        else:
            return data, True

    except Exception as e:
        logging.error(f'Error in regression_analysis: {e}')
        return False


data_analysis(f500_clean)
