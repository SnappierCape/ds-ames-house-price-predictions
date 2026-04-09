# =============================================================================
# DATA PROCESSSING MODULE
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import random as rnd
from typing import Any, Union, List, Dict, Optional

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rnd.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def df_aggregate(
    df: pd.DataFrame,
    value_col: str,
    labels: str,
    threshold: float = 3.0,
    rows_exclude: Union[List[Any], Any] = []
) -> pd.DataFrame:
    """
    Aggregates rows with values below a certain threshold into an 'Other' category,
    while protecting specific rows from being grouped.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    value_col : str
        The column used to evaluate the threshold for aggregation.
    labels : str
        The column name identifying the labels (or the name of the index).
    threshold : float, default 3.0
        Values in 'value_col' below this number will be aggregated.
    rows_exclude : list or Any, default []
        Specific index values that should never be aggregated into 'Other', 
        regardless of their value.

    Returns
    -------
    pd.DataFrame
        A DataFrame with small values condensed into an 'Other' row.
    """
    # Normalize the "rows_exclude" variable to a list
    if not isinstance(rows_exclude, list):
        rows_exclude = [rows_exclude]

    # Check if the labels are in the index
    labels_in_index = all(labels == df.index)

    # Exclude the requested rows (if they exist)
    if any(item in df.index for item in rows_exclude):
        excluded_rows = df.loc[rows_exclude]
        df = df.drop(index=rows_exclude)
    else:
        print(
            'Error!!! The selected "rows_exclude" argument has no correspondence in the dataframe index.'
        )
        excluded_rows = pd.Series()

    # If the labels are in the index, move them to a column and reset the index
    if labels_in_index:
        df["labels"] = df.index
        df.index = np.arange(0, len(df))

    # Find small rows, large rows, categorical columns and numerical columns
    small_rows = df[df[value_col] < threshold]
    large_rows = df[df[value_col] >= threshold]
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    # If there are rows under the threshold, aggregate them
    if len(small_rows) >= 2:
        other_row = pd.DataFrame()
        for col in df.columns:
            if col in num_cols:
                other_row[col] = [small_rows[col].sum()]
            else:
                other_row[col] = "Other"

        # Aggregate large rows and other row
        aggregated_df = pd.concat(
            [large_rows, other_row],
            ignore_index=False,
        )

    else:
        aggregated_df = df

    # Put the original index in place (if it contained the labels)
    if labels_in_index:
        aggregated_df = aggregated_df.set_index("labels")

    # Re-attach the "excluded_rows" if present
    if not excluded_rows.empty:
        aggregated_df = pd.concat(
            [aggregated_df, excluded_rows],
            ignore_index=False,
        )
    # At this point i have the original index in place, so if it is numeric it has been summed
    # in the "other_row", so I have to reset it
    if pd.api.types.is_any_real_numeric_dtype(aggregated_df.index):
        aggregated_df.index = np.arange(0, len(aggregated_df))

    return aggregated_df


def create_freq_table(df: pd.DataFrame, column: str, round_digit: int = 5) -> pd.DataFrame:
    """
    Generates a frequency distribution table for a categorical variable, 
    including absolute and relative frequencies.

    Parameters
    ----------
    df : pd.DataFrame
        The source DataFrame.
    column : str
        The categorical column to analyze.
    round_digit : int, default 5
        Decimal precision for the percentage calculations.

    Returns
    -------
    pd.DataFrame
        A table indexed by category with 'abs_freq', 'rel_freq_%', and a 'NaN' row 
        if missing values exist.
    """

    # Obtain absolute and relative frequencies
    abs_freq = df[column].value_counts()
    rel_freq = round(
        (abs_freq / df[column].size) * 100,
        round_digit,
    )

    # Create the dataframe to display
    freq_table = pd.DataFrame(
        {
            "abs_freq": abs_freq,
            "rel_freq_%": rel_freq,
        }
    )

    # Sort the table in descending order of absolute frequency
    freq_table = freq_table.sort_values(by="abs_freq", ascending=False)

    # Calculate the missing values row
    na_row = pd.DataFrame(
        {
            "abs_freq": [df[column].isna().sum()],
            "rel_freq_%": [(df[column].isna().sum() / len(df)) * 100],
        },
        index=["NaN"],
    )

    # Add the missing values row to the frequency table if not empty
    if all(na_row["abs_freq"]) != 0:
        freq_table = pd.concat([freq_table, na_row])

    return freq_table


def drop_na_columns(df, threshold=100):
    """
    Drops the columns of a dataframe that contains more than a certain number of empty values.

    Parameters:
    - df (pd.DataFrame): The input dataframe from which the function removes columns.
    - threshold (int): The maximum number of empty values before a column gets dropped.

    Returns:
    - df (pd.DataFrame): The dataframe without the columns that have beed dropped
    """

    # Drop columns
    dropped_cols = pd.Series()
    for col in df.columns:
        if df[col].isna().sum() > threshold:
            df = df.drop(col, axis=1)
            dropped_cols = pd.concat([dropped_cols, pd.Series(col)])

    # Display which columns have been dropped
    print(f"Dropped columns:\n{dropped_cols}")

    # Return the clean dataframe
    return df


def normalize(array: Union[pd.Series, np.ndarray], min_val: float = 0.0, max_val: float = 1.0) -> Union[pd.Series, np.ndarray]:
    """
    Performs Min-Max scaling on an array to transform values into a specific range.

    Parameters
    ----------
    array : pd.Series or np.ndarray
        The numeric data to scale.
    min_val : float, default 0.0
        The desired minimum value.
    max_val : float, default 1.0
        The desired maximum value.

    Returns
    -------
    Union[pd.Series, np.ndarray]
        The scaled data. Note: May result in NaNs if the input array is constant.
    """
    array_range = array.max() - array.min()
    normalized_range = max - min
    normalized_array = (
        ((array - array.min()) / array_range)  # Normalize to [0, 1]
        * normalized_range  # Scale to the desired interval
        + min  # Add min to start from the desired value
    )

    return normalized_array

# -----------------------------------------------------------------------------
# Function "one_hot_encoding()" with helpers.
# -----------------------------------------------------------------------------

def _encode_variable(var: pd.Series, ref_mod: Any) -> pd.DataFrame:
    """HELPER
    Creates binary dummy variables for a single categorical Series, 
    omitting the specified reference modality.

    Parameters
    ----------
    var : pd.Series
        The categorical column to encode.
    ref_mod : Any
        The specific category value to treat as the baseline (dropped).

    Returns
    -------
    pd.DataFrame
        A DataFrame of binary columns, one for each category except the reference.
    """
    
    # Find the unique modalities.
    uniq_mods = pd.unique(var)
    uniq_mods = uniq_mods[uniq_mods != ref_mod]

    # Find the name of the variable
    var_name = var.name

    # Build the binary columns
    one_hot_var = pd.DataFrame()
    one_hot_var.index = np.arange(len(var))

    for mod in uniq_mods:
        one_hot_var[f"{var_name}_{mod}"] = (var == mod).astype(int)

    return one_hot_var


def one_hot_encoding(
    df: pd.DataFrame, 
    vars: Union[List[str], str], 
    ref_mod: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """MAIN
    Applies One-Hot Encoding (OHE) to multiple columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    vars : list of str or str
        The column(s) to be encoded.
    ref_mod : dict, optional
        A mapping of {column_name: reference_value}. If a variable is missing 
        from this dict, the most frequent category (mode) is used as the reference.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with categorical columns replaced by dummy variables.
    """
    # Standardize "vars" as a list.
    if not isinstance(vars, list):
        vars = [vars]

    # Initialize the result df.
    one_hot_df = df
    one_hot_df = one_hot_df.drop(columns=df.columns[df.columns.isin(vars)])

    # Loop through the variables to encode.
    for var in vars:
        if ref_mod and var in ref_mod:
            ref_value = ref_mod[var]
        else:
            ref_value = df[var].value_counts().idxmax()
            print(f'Auto-selected reference value for "{var}": {ref_value}')
        one_hot_var = _encode_variable(df[var], ref_value)
        one_hot_df = pd.concat([one_hot_df, one_hot_var], axis=1)

    return one_hot_df

# -----------------------------------------------------------------------------
# Other functions.
# -----------------------------------------------------------------------------

def ordinal_encoding(df: pd.DataFrame, encoding_map: Dict[str, Dict[Any, Any]]) -> pd.DataFrame:
    """
    Transforms categorical levels into ordered integers based on a provided map.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    encoding_map : dict of dicts
        A dictionary where keys are column names and values are dictionaries 
        mapping {category: integer_value}.

    Returns
    -------
    pd.DataFrame
        The DataFrame with specified columns converted to ordinal integers.
    """
    for var, mapping in encoding_map.items():
        if var not in df.columns:
            print(f'Warning: "{var}" not found in df, skipping...')
            continue
        df[var] = df[var].map(mapping)

        # See if there are unmapped categories.
        unmapped = df[var].isna().sum()
        if unmapped > 0:
            print(
                f'Warning: "{var}" has {unmapped} unmapped modalities --> NaN.'
            )

    return df
