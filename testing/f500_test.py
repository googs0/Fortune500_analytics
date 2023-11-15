import pytest
import pandas as pd
import matplotlib
import unittest
from unittest.mock import MagicMock, patch

matplotlib.use('agg')

from main_4 import (
    download_and_load_data,
    clean_data,
    validate_data,
    handle_outliers,
    regression_analysis,
    correlation_matrix,
    cluster_analysis,
    time_series_analysis,
    pair_plot,
    box_plot,
    plot_scatter,
    data_analysis,
)


@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    data = {
        'Year': [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033],
        'Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'Company': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
        'Profit': [100, 150, 80, 200, 120, 150, 200, 250, 130, 220, 300, 350],
        'Revenue': [1000, 1200, 800, 1500, 2000, 2500, 1300, 600, 1200, 1500, 2000, 2200],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mocker_show_close():
    # Mock plt.show() and plt.close() to avoid displaying the plot during testing
    with patch('matplotlib.pyplot.close'):
        yield


def test_download_and_load_data(sample_data, monkeypatch):
    # Mock the requests.get function to return sample_data
    monkeypatch.setattr('requests.get', lambda *args, **kwargs: MagicMock(content=sample_data.to_csv(index=False)))

    # Call the function and check if it returns a DataFrame
    result = download_and_load_data('mocked_url', 'mocked_filename')
    assert isinstance(result, pd.DataFrame)


def test_clean_data(sample_data):
    # Test the clean_data function
    result = clean_data(sample_data)

    # Check if the returned result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Add more specific tests for your cleaning logic if needed


def test_validate_data(sample_data, caplog):
    # Test the validate_data function
    validate_data(sample_data)

    # Check if a warning message is logged for invalid data types
    assert 'Warning: Data type of column' in caplog.text


def test_handle_outliers(sample_data):
    # Test the handle_outliers function
    result, outliers = handle_outliers(sample_data, 'Profit')

    # Check if the returned result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if outliers are a DataFrame
    assert isinstance(outliers, pd.DataFrame)


def test_regression_analysis(sample_data, capsys):
    # Mock plt.show() to avoid displaying the plot during testing
    with unittest.mock.patch('matplotlib.pyplot.show'):
        # Test the regression_analysis function
        result = regression_analysis(sample_data, ['Year'], 'Profit', degree=2, include_outliers=True)

        # Check if the function executed successfully
        assert result, 'Regression analysis failed.'

        # Unpack the result
        data, outliers, result_flag = result

        # Check if result_flag is a boolean (success_flag)
        assert isinstance(result_flag, bool), 'Invalid value for success_flag.'

        # Check if outliers were returned
        assert 'outliers' in locals(), 'Outliers were not returned.'

        # Capture printed output to check if the results are printed
        captured = capsys.readouterr()
        assert 'Linear Regression Results:' in captured.out
        assert 'Polynomial Regression Results:' in captured.out


def test_correlation_matrix(sample_data, mocker_show_close):
    # Mock plt.show() to avoid displaying the plot during testing
    with unittest.mock.patch('matplotlib.pyplot.show'):
        # Test the correlation_matrix function
        result = correlation_matrix(sample_data)
        assert result is True


def test_cluster_analysis(sample_data):
    # Mock plt.show() to avoid displaying the plot during testing
    with unittest.mock.patch('matplotlib.pyplot.show'):
        # Test the cluster_analysis function
        result = cluster_analysis(sample_data)

        # Check if the function executed successfully
        assert result is True


def test_time_series_analysis(sample_data):
    # Mock plt.show() to avoid displaying the plot during testing
    with unittest.mock.patch('matplotlib.pyplot.show'):
        # Test the time_series_analysis function with a smaller period
        result = time_series_analysis(sample_data, decomposition_model='additive', period=2)

        # Check if the function executed successfully
        assert result is True


def test_pair_plot(sample_data):
    # Mock plt.show() to avoid displaying the plot during testing
    with unittest.mock.patch('matplotlib.pyplot.show'):
        # Test the pair_plot function
        result = pair_plot(sample_data)

        # Check if the function executed successfully
        assert result is True


def test_box_plot(sample_data):
    # Mock plt.show() to avoid displaying the plot during testing
    with unittest.mock.patch('matplotlib.pyplot.show'):
        # Test the box_plot function
        result = box_plot(sample_data)

        # Check if the function executed successfully
        assert result is True


def test_plot_scatter(sample_data):
    # Mock plt.show() to avoid displaying the plot during testing
    with unittest.mock.patch('matplotlib.pyplot.show'):
        # Test the plot_scatter function
        result = plot_scatter(sample_data)

        # Check if the function executed successfully
        assert result is True


def test_data_analysis(sample_data, capsys):
    # Mock plt.show() to avoid displaying the plot during testing
    with unittest.mock.patch('matplotlib.pyplot.show'):
        # Test the data_analysis function
        result_data = data_analysis(sample_data)

        # Check if the function executed successfully
        assert isinstance(result_data, pd.DataFrame) and 'Cluster' in result_data.columns

        # Capture printed output to check if the results are printed
        captured = capsys.readouterr()
        assert 'Data from' in captured.out
        assert 'Rows in the dataset' in captured.out
