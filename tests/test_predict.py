import pytest
import pandas as pd
import numpy as np  # <-- Add this import at the top
from src.data_utils import add_two_numbers, handle_missing_values # <-- Add new function
from src.data_utils import add_two_numbers, handle_missing_values, predict_simple_linear # <-- Add new function
# ... (fixture 'sample_dataframe' and 'test_add_two_numbers_param' are here) ...


@pytest.mark.parametrize("input_val, expected_output", [
    (0, 1),
    (1, 3),
    (5, 11),
    (-2, -3)
])
def test_predict_simple_linear(input_val, expected_output):
    """Test the simple linear prediction function with various inputs."""
    result = predict_simple_linear(input_val)
    assert result == expected_output

@pytest.fixture
def dataframe_with_missing():
    """Provides a DataFrame with missing values for testing."""
    data = {'numeric_col': [1, 2, np.nan, 4, 5], 'categorical_col': ['A', 'B', 'C', 'D', 'E']}
    df = pd.DataFrame(data)
    return df

@pytest.mark.skip(reason="This feature is temporarily disabled for rework.")
def test_a_new_feature():
    # ... test code for a feature that isn't ready ...
    assert False

@pytest.mark.xfail(reason="Known bug in the external data source library.")
def test_connection_to_buggy_api():
    # ... code that you know will fail ...
    raise ConnectionError("API connection failed")    


def test_handle_missing_values_mean(dataframe_with_missing):
    """Test handling missing values using the 'mean' strategy."""
    result_df = handle_missing_values(dataframe_with_missing, strategy='mean')
    # Expected mean of numeric_col (1+2+4+5)/4 = 3.0
    assert result_df['numeric_col'].isnull().sum() == 0
    assert result_df.loc[2, 'numeric_col'] == 3.0


def test_handle_missing_values_drop(dataframe_with_missing):
    """Test handling missing values using the 'drop' strategy."""
    result_df = handle_missing_values(dataframe_with_missing, strategy='drop')
    assert result_df.shape[0] == 4 # One row with a NaN should be dropped
    assert result_df['numeric_col'].isnull().sum() == 0


def test_handle_missing_values_invalid_strategy(dataframe_with_missing):
    """Test that an error is raised for an invalid strategy."""
    with pytest.raises(ValueError, match="Unknown strategy: invalid"):
        handle_missing_values(dataframe_with_missing, strategy='invalid')


# ... (tests 'test_dataframe_columns' and 'test_dataframe_shape' are here) ...
