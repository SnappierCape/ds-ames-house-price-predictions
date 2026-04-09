# =============================================================================
# NUMERICAL AND VISUAL ANALYSIS MODULE
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import random as rnd
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .processing import create_freq_table, df_aggregate

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rnd.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def numerical_variable_analysis(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Analyzes a numerical variable by providing descriptive statistics and a 
    visual representation via a violin plot with an embedded boxplot.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing the variable to analyze.
    column : str
        The name of the numerical column to be analyzed.

    Returns
    -------
    summary : pd.DataFrame
        A summary table containing key statistical measures including missing 
        counts, central tendency, dispersion, and shape of the distribution.

    Figures
    -------
    A violin plot with an embedded boxplot, mean line, and median line, 
    including a text box summarizing key statistics.
    """

    # Define the data we will be working with:
    data = df[
        column
    ]  # This will be used just to display the missing counts in the plot
    data_clean = (
        data.dropna()
    )  # This will be used for the statistical calculations and the plot itself

    # Create the Figure and the Axes
    fig, ax = plt.subplots(figsize=(7, 10))

    # Create the Violin Plot
    violin = ax.violinplot(
        data_clean,
        positions=[1],
        widths=[0.6],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Customize the Violin's Body and Style
    for body in violin["bodies"]:
        (body.set_color("lightblue"),)
        body.set_alpha(0.8)

    # Overlay a Boxplot
    boxplot = ax.boxplot(
        data_clean,
        positions=[1],
        widths=[0.15],
        patch_artist=True,
        showfliers=True,
        flierprops=dict(
            marker="o",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=4,
            alpha=0.35,
        ),
        boxprops=dict(facecolor="none", color="black"),
        medianprops=dict(
            color="none", linewidth=0
        ),  # Remove the default Median Line
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    # Add the Mean Line
    # Calculate the mean
    mean = np.mean(data_clean)

    # Line
    ax.hlines(
        mean,
        xmin=1 - 0.3,
        xmax=1 + 0.3,
        color="blue",
        linewidth=1.2,
        linestyle="--",
    )

    # Text
    ax.text(
        x=1 - 0.35,
        y=mean,
        s="Mean",
        color="blue",
        verticalalignment="center",
        horizontalalignment="right",
    )

    # Add the Median Line
    # Calculate median
    median = np.median(data_clean)

    # Line
    ax.hlines(
        median,
        xmin=1 - 0.3,
        xmax=1 + 0.3,
        color="green",
        linewidth=1.2,
        linestyle="--",
    )

    # Text
    ax.text(
        x=1 + 0.35,
        y=median,
        s="Median",
        color="green",
        verticalalignment="center",
        horizontalalignment="left",
    )

    # Add the Statistics Text Box
    # Key Statistics calculations
    miss_count = data.isna().sum()
    miss_prop = (miss_count / len(data)) * 100
    mode_result = sp.stats.mode(data_clean)
    mode = mode_result[0]
    variance = np.var(data_clean)
    stdev = np.std(data_clean)
    skewness = sp.stats.skew(data_clean)
    kurtosis = sp.stats.kurtosis(data_clean)
    min_val = np.min(data_clean)
    max_val = np.max(data_clean)
    q1 = np.percentile(data_clean, 25)
    q3 = np.percentile(data_clean, 75)
    iqr = q3 - q1
    outliers = (data > q3 + 1.5 * iqr) | (data < q1 - 1.5 * iqr)
    n_outliers = outliers.sum()

    # Create the text box for the statistics
    stats_textbox = (
        f"Key Statistical Measures:\n"
        f"Miss Count:    {miss_count:,.0f}\n"
        f"Miss Prop:      {miss_prop:,.2f}%\n"
        f"Mean:            {mean:,.2f}\n"
        f"Median:         {median:,.2f}\n"
        f"Mode:            {mode:,.2f}\n"
        f"Min:               {min_val:,.2f}\n"
        f"Q1:                {q1:,.2f}\n"
        f"Q3:                {q3:,.2f}\n"
        f"Iqr:                {iqr:,.2f}\n"
        f"Max:              {max_val:,.2f}\n"
        f"N° outliers:    {n_outliers:,.0f}\n"
        f"Variance:       {variance:,.2f}\n"
        f"StdDev:         {stdev:,.2f},\n"
        f"Skewness:     {skewness:,.2f}\n"
        f"Kurtosis:        {kurtosis:,.2f}"
    )

    # Add the text box to the plot
    ax.text(
        x=0.05,
        y=0.95,
        s=stats_textbox,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="square, pad = 0.7",
            facecolor="white",
            alpha=1,
            edgecolor="black",
        ),
    )

    # Add Title and Labels
    ax.set_title(
        f'Density Plot for the "{data_clean.name}" Variable',
        fontsize=15,
        color="navy",
    )
    ax.set_ylabel(
        f"{data_clean.name}",
        fontsize=12,
        color="darkgreen",
    )
    ax.set_xlabel(
        "Density of Observations",
        fontsize=12,
        color="darkgreen",
    )

    # Generate output table with variable's key statistic
    summary = pd.DataFrame(
        {
            f"{column}": [
                miss_count,
                miss_prop,
                mean,
                median,
                mode,
                variance,
                stdev,
                skewness,
                kurtosis,
                min_val,
                max_val,
                q1,
                q3,
                iqr,
                n_outliers,
            ]
        }
    )

    # Add index
    summary.index = [
        "miss_count",
        "miss_prop",
        "mean",
        "median",
        "mode",
        "variance",
        "stdev",
        "skewness",
        "kurtosis",
        "min",
        "max",
        "q1",
        "q3",
        "iqr",
        "n_outliers",
    ]

    # Return
    return summary


def nested_donut_plot(freq_table: pd.DataFrame, var_name: str) -> None:
    """
    Plots a nested donut chart displaying both absolute and relative frequencies.

    Parameters
    ----------
    freq_table : pd.DataFrame
        DataFrame containing 'abs_freq' and 'rel_freq_%' columns, with class 
        labels assigned to the index.
    var_name : str
        The name of the variable under analysis, used for the plot title.

    Returns
    -------
    None

    Figures
    -------
    A nested donut chart visualizing absolute counts and percentage distributions 
    along with an informative legend.
    """

    # Create Figure and Axes
    fig, ax = plt.subplots(figsize=(12, 12))

    # Create custom color Palette
    colors = ["midnightblue", "lightskyblue"]
    color_palette = LinearSegmentedColormap.from_list("custom_gradient", colors, N=256)

    # Create the color map
    color_map = color_palette(np.linspace(0, 1, len(freq_table))).tolist()

    # Set the "Other" row to be orange (if it exist)
    if "Other" in freq_table.index:
        other_pos = freq_table.index.get_loc("Other")
        color_map[other_pos] = "orange"

    # Set the "NaN" row to be grey (if it exist)
    if "NaN" in freq_table.index:
        nan_pos = freq_table.index.get_loc("NaN")
        color_map[nan_pos] = "dimgray"

    # Create inner dunut for percentages
    int_wedges, int_texts, int_autotexts = ax.pie(
        freq_table["rel_freq_%"],
        startangle=90,
        radius=0.7,
        wedgeprops=dict(
            width=0.22,
            edgecolor="white",
            linewidth=4,
        ),
        colors=color_map,
        autopct="%1.1f%%",
        pctdistance=1.1,
    )

    # Add percentage labels
    for int_autotext in int_autotexts:
        (int_autotext.set_color("black"),)
        int_autotext.set_fontsize(11)

    # Add outer donut for absolute frequencies
    ext_wedges, ext_texts, ext_autotexts = ax.pie(
        freq_table["abs_freq"],
        startangle=90,
        radius=1.15,
        wedgeprops=dict(
            width=0.22,
            edgecolor="white",
            linewidth=4,
        ),
        colors=color_map,
        autopct="%1.0f",
    )

    # Remove the absolute value labels
    for ext_autotext in ext_autotexts:
        ext_autotext.set_text("")

    # Add the class names labels
    for wedge, label in zip(ext_wedges, freq_table.index):
        # Get the start, center, and end angle in radiants
        theta1 = np.deg2rad(wedge.theta1)
        theta2 = np.deg2rad(wedge.theta2)
        center = (theta1 + theta2) / 2

        # Use the center angle to calculate the x and y position of the label
        x = 1.28 * np.cos(center)
        y = 1.28 * np.sin(center)

        # Add the actual label to the chart
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="light",
        )

    # Add title
    ax.set_title(
        f'Absolute and Relative Frequencies for the "{var_name}" Variable',
        fontsize=16,
        fontweight="bold",
        pad=25,
        color="darkblue",
    )

    # Add legend
    legend_labels = [
        f"{label}: {freq:.0f}"
        for label, freq in zip(
            freq_table.index,
            freq_table["abs_freq"],
        )
    ]

    ax.legend(
        legend_labels,
        title="Absolute Frequencies",
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )


def categorical_variable_analysis(df: pd.DataFrame, column: str, threshold: int | float = 3) -> None:
    """
    Analyzes a categorical variable by generating a frequency table and a 
    nested donut chart, with low-frequency categories grouped into 'Other'.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing the categorical variable.
    column : str
        The name of the categorical column to be analyzed.
    threshold : int | float, default 3
        Percentage threshold below which categories are aggregated into 'Other'.

    Returns
    -------
    None

    Figures
    -------
    A nested donut chart illustrating the distribution of the categorical variable 
    after processing and aggregation.
    """

    # Create the frequency table
    freq_table = create_freq_table(df, column)

    # Aggregate the frequency table based on the threshold
    freq_table = df_aggregate(
        freq_table,
        value_col="rel_freq_%",
        threshold=threshold,
        labels=freq_table.index,
        rows_exclude="NaN",
    )

    # Plot the nested donut chart
    nested_donut_plot(freq_table, column)


def corr_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs pairwise correlation analysis between numeric columns using the 
    Pearson correlation coefficient.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing numeric variables to analyze.

    Returns
    -------
    corr_matrix : pd.DataFrame
        A symmetric matrix containing Pearson correlation coefficients for all 
        numeric variable pairs.
    corr_table : pd.DataFrame
        A flattened and sorted table of unique variable pairs, ordered by the 
        absolute value of their correlation coefficients.
    """

    # Correlation matrix
    num_cols = df.select_dtypes(include=np.number).columns
    corr_matrix = df[num_cols].corr()

    # Extract the shape of the correlation matrix
    corr_matrix_shape = corr_matrix.shape

    # Create a matrix filled with "1" with the same shape
    ones_matrix = np.ones(corr_matrix_shape)

    # Select only the upper triangle excludint the elements on the diagonal
    upper_triangle = np.triu(ones_matrix, k=1)

    # Convert the values to booleans
    upper_triangle = upper_triangle.astype(bool)

    # Use the boolean triangle to select values from the "corr_matrix"
    corr_triangle = corr_matrix.where(upper_triangle)

    # Stack the triangle to create a correlation table
    corr_series = corr_triangle.stack().reset_index()
    corr_series.columns = [
        "var1",
        "var2",
        "corr_coeff",
    ]

    # Sort values
    corr_table = corr_series.sort_values(
        "corr_coeff", key=abs, ascending=False
    )

    # Return
    return corr_matrix, corr_table


def corr_heatmap(df: pd.DataFrame) -> None:
    """
    Generates and displays a correlation heatmap for the numeric variables 
    within a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset to visualize.

    Returns
    -------
    None

    Figures
    -------
    A color-coded heatmap representing the Pearson correlation matrix, 
    utilizing a diverging colormap with a colorbar indicator.
    """

    # Create correlation matrix
    corr_matrix, corr_table = corr_analysis(df)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 9))

    # Create color map
    cmap = plt.get_cmap("bwr")
    norm = plt.Normalize(vmin=-1, vmax=1)

    # Plot the heatmap
    im = ax.imshow(corr_matrix, cmap=cmap, norm=norm)

    # Add colorbar
    cbar = ax.figure.colorbar(
        im,
        ax=ax,
        shrink=0.9,
        aspect=30,
        pad=0.025,
    )

    cbar.ax.set_ylabel(
        "Pearson's Correlation Coefficient",
        rotation=90,
        va="top",
        ha="center",
        fontsize=14,
        labelpad=18,
    )

    # Set axis ticks and labels
    ax.set_xticks(np.arange(0, len(corr_matrix.columns)))
    ax.set_yticks(np.arange(0, len(corr_matrix)))
    ax.set_xticklabels(
        corr_matrix.columns,
        rotation=90,
        ha="center",
    )
    ax.set_yticklabels(corr_matrix.index, rotation=0)

    # Add title and grid
    ax.set_title("Correlation Heatmap", fontsize=17, pad=15)


def mcar_chi2_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes a series of Chi-square tests of independence to evaluate if missing 
    values in specific columns are Completely At Random (MCAR).

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing variables with missing values and 
        categorical predictors.

    Returns
    -------
    results : pd.DataFrame
        A summary table listing pairwise combinations of predictor variables and 
        missingness indicators, along with their associated p-values.
    """

    # Ensure the dataframe contains only categorical variables
    num_cols_count = len(df.select_dtypes(exclude="O").columns)
    if num_cols_count >= 1:
        print(
            f"Warning: {num_cols_count} numerical variables found in the dataframe, this function will select only the categorical ones as predictors, so the numerical ones will be excluded."
        )

    # Find out which columns contain missing values
    cols_with_missing = df.columns[df.isna().any()].to_list()

    # Create dataframe with missing indicators
    missing_indicators = pd.DataFrame()
    for col in cols_with_missing:
        missing_indicators[col] = df[col].isna().astype(int)

    # Set up the result array
    p_values = np.array([])
    pred_vars = np.array([])
    miss_vars = np.array([])

    # Select categorical columns to use as predictors
    X = df.select_dtypes(include="O")

    # Replace the missing values in the predictors with simple "NaN" strings to trick the system into
    # believing that they are another category. I don't know why but if i don't do this i get an error.
    X = X.fillna("NaN")

    # Set up the double nested for cycle
    # First level
    for col_with_missing in cols_with_missing:  # This level cycles through only the columns that contain missing values
        y = missing_indicators[col_with_missing]

        # Remove y from the predictors in every cycle
        if col_with_missing in X.columns:
            X = X.drop(columns=col_with_missing)

        # Second level
        for x in X.columns:  # This level cycles through the predictors excluding the variable that corresponds to the current step of the parent cycle
            # Create contingency table
            contingency_table = pd.crosstab(X[x], y)

            # Run the chi-square test of independence
            # Observed values
            obs = np.array(
                contingency_table
            )  # I prefer to work with NumPy arrays in this case
            row_tot = obs.sum(axis=1)
            col_tot = obs.sum(axis=0)
            grand_tot = sum(row_tot)

            # Expected values
            exp = (
                np.outer(row_tot, col_tot) / grand_tot
            )  # The "outer()" method multiplies every item of an array with every item of the other array

            # Chi-square statistic
            chisq = np.sum(((obs - exp) ** 2) / exp)

            # Degrees of freedom
            deg_free = (obs.shape[0] - 1) * (obs.shape[1] - 1)

            # P-value
            p_value = 1 - sp.stats.chi2.cdf(chisq, deg_free)
            p_value

            # Add results
            # P-values
            p_values = np.append(p_values, p_value)

            # Predictors variables
            pred_vars = np.append(pred_vars, x)

            # Missing variable
            miss_vars = np.append(miss_vars, y.name)

    # Create result dataframe
    results = pd.DataFrame(
        {
            "Prediction Variable": pred_vars,
            "Missing Variable": miss_vars,
            "P-Value": p_values,
        }
    )

    # Sort result dataframe
    results = results.sort_values(by="P-Value", ascending=False)

    # Return
    return results


def df_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a comprehensive statistical summary of the dataset, partitioned 
    by numerical and categorical variable types.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset to be summarized.

    Returns
    -------
    num_results : pd.DataFrame
        A summary table for numerical variables including measures of central 
        tendency, dispersion, shape, and outlier counts.
    cat_results : pd.DataFrame
        A summary table for categorical variables including modality counts, 
        mode identification, and frequency metrics.
    """

    # Divide numerical variables from categorical variables
    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=np.number)

    # Create clean dataframes without missing values
    num_df_clean = num_df.dropna()
    cat_df_clean = cat_df.dropna()

    # Summary for numerical variables
    # Key Statistics calculations
    mean = num_df_clean.mean(axis=0)
    median = num_df_clean.median(axis=0)
    miss_count = num_df.isna().sum(axis=0)
    miss_prop = (miss_count / len(num_df)) * 100
    mode_result = sp.stats.mode(num_df_clean, axis=0)
    mode_count = mode_result[1]
    mode = mode_result[0]
    variance = np.var(num_df_clean, axis=0)
    stdev = np.std(num_df_clean, axis=0)
    skewness = sp.stats.skew(num_df_clean, axis=0)
    kurtosis = sp.stats.kurtosis(num_df_clean, axis=0)
    min_val = np.min(num_df_clean, axis=0)
    max_val = np.max(num_df_clean, axis=0)
    q1 = np.percentile(num_df_clean, 25, axis=0)
    q3 = np.percentile(num_df_clean, 75, axis=0)
    iqr = q3 - q1

    # Outliers
    # This part is tricky, good luck understanding why it is the way it is
    outliers = pd.DataFrame()
    for col in enumerate(num_df_clean.columns):
        col_outliers = (
            num_df_clean.iloc[:, col[0]] > q3[col[0]] + 1.5 * iqr[col[0]]
        ) | (num_df_clean.iloc[:, col[0]] < q1[col[0]] - 1.5 * iqr[col[0]])
        outliers[num_df_clean.iloc[:, col[0]].name] = col_outliers

    # Count outliers
    outliers_count = outliers.sum(axis=0)

    # Creating result dataframe for numerical variables
    num_results = pd.DataFrame(
        {
            "Mean": mean,
            "Median": median,
            "Missing Count": miss_count,
            "Missing Prop": miss_prop,
            "Mode": mode,
            "Mode Count": mode_count,
            "Variance": variance,
            "StDev": stdev,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "Min": min_val,
            "Max": max_val,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "N° Outliers": outliers_count,
        }
    )

    # Transpose the dataframe
    num_results = num_results.T

    # Add a name to the index
    num_results.reset_index(inplace=False)
    num_results.columns.name = "Key Statistics"

    # Summary for categorical variables
    # Initializing all lists
    modalities_count = np.array([])
    mode = np.array([])
    mode_count = np.array([])
    miss_count = np.array([])

    # Fill them for every variable
    for col in cat_df_clean.columns:
        modalities_count = np.append(
            modalities_count,
            len(cat_df_clean[col].value_counts()),
        )
        mode = np.append(
            mode,
            pd.DataFrame(cat_df_clean[col].value_counts()).iloc[0].name,
        )
        mode_count = np.append(
            mode_count,
            pd.DataFrame(cat_df_clean[col].value_counts()).iloc[0]["count"],
        )
        miss_count = np.append(miss_count, cat_df[col].isna().sum())

    miss_prop = (miss_count / len(cat_df)) * 100

    # Creating results dataframe for categorical variables
    cat_results = pd.DataFrame(
        {
            "N° Modalities": modalities_count,
            "Mode": mode,
            "Mode Count": mode_count,
            "Missing Count": miss_count,
            "Missing Prop": miss_prop,
        }
    )

    # Transpose the dataframe
    cat_results = cat_results.T

    # Add a name to the index
    cat_results.reset_index(inplace=False)
    cat_results.columns = cat_df_clean.columns
    cat_results.columns.name = "Key Statistics"

    return num_results, cat_results

# -----------------------------------------------------------------------------
# Function "scatter_2d()" with helpers.
# -----------------------------------------------------------------------------

def _normalize(array: pd.Series | np.ndarray, out_min: float = 0.0, out_max: float = 1.0) -> pd.Series:
    """HELPER
    Scales an input array to a specified output range [out_min, out_max].

    Parameters
    ----------
    array : pd.Series or np.ndarray
        The numeric data to be scaled.
    out_min : float, default 0.0
        The minimum value of the target range.
    out_max : float, default 1.0
        The maximum value of the target range.

    Returns
    -------
    pd.Series
        The normalized array. If the input range is zero, returns the midpoint 
        of the target range.
    """
    arr_min = array.min()
    arr_max = array.max()
    arr_range = arr_max - arr_min
    if arr_range == 0:
        # Constant column: return midpoint of output range for all values
        return pd.Series([(out_min + out_max) / 2] * len(array))
    return out_min + (array - arr_min) / arr_range * (out_max - out_min)


def _generate_hex_colors(n: int = 40) -> list[str]:
    """HELPER
    Generates a list of random hexadecimal color strings.

    Parameters
    ----------
    n : int, default 40
        The number of unique hex colors to generate.

    Returns
    -------
    list of str
        A list of n strings representing hexadecimal colors.
    """
    np.random.seed(42)
    return ["#{:06x}".format(np.random.randint(0, 0xFFFFFF)) for _ in range(n)]


def _resolve_aesthetic(value: Any, df: pd.DataFrame, kind: str) -> dict[str, Any]:
    """HELPER
    Resolves a visual aesthetic parameter into per-point data and metadata.

    Parameters
    ----------
    value : Any
        The user input for the aesthetic (column name, scalar, or None).
    df : pd.DataFrame
        The source dataset for column-based mapping.
    kind : {"color", "size", "alpha", "shape"}
        The type of aesthetic being resolved.

    Returns
    -------
    dict
        A dictionary containing the resolved 'array' of values, a 'legend' flag, 
        and specific metadata (e.g., colormaps, categories) for building legends.
    """

    # ── Is it a column reference? ──────────────────────────────────────────────
    is_column = isinstance(value, str) and (value in df.columns)

    # ── COLOR ──────────────────────────────────────────────────────────────────
    if kind == "color":
        if is_column:
            col = df[value]
            categories = np.unique(col)
            hex_colors = _generate_hex_colors(len(categories))
            # Map each category to a random float, then use a ListedColormap
            rng_vals = np.random.rand(len(categories))
            cat_to_float = dict(zip(categories, rng_vals))
            return {
                "legend": True,
                "array": col.map(cat_to_float),  # floats for cmap
                "cmap": ListedColormap(hex_colors),
                "categories": categories,
                "hex_colors": hex_colors,
                "col_name": value,
            }

        # Fixed color string (e.g. "blue", "#ff0000") or None → no legend
        fixed = value if isinstance(value, str) else "steelblue"
        return {
            "legend": False,
            "array": pd.Series([fixed] * len(df)),
            "cmap": None,
        }

    # ── SIZE ───────────────────────────────────────────────────────────────────
    elif kind == "size":
        if is_column:
            col = df[value]
            scaled = _normalize(col, out_min=4, out_max=1000)
            return {
                "legend": True,
                "array": scaled,
                "raw_col": col,
                "col_name": value,
            }

        fixed = value if isinstance(value, (int, float)) else 30
        return {
            "legend": False,
            "array": pd.Series([fixed] * len(df)),
        }

    # ── ALPHA ──────────────────────────────────────────────────────────────────
    elif kind == "alpha":
        if is_column:
            col = df[value]
            scaled = _normalize(col, out_min=0.1, out_max=1.0)
            return {
                "legend": True,
                "array": scaled,
                "col_name": value,
            }

        fixed = value if isinstance(value, (int, float)) else 0.3
        return {
            "legend": False,
            "array": pd.Series([fixed] * len(df)),
        }

    # ── SHAPE ──────────────────────────────────────────────────────────────────
    elif kind == "shape":
        if is_column:
            col = df[value]
            categories = np.unique(col)
            # Cycle through the default marker list if there are more categories
            default_markers = [
                "o",
                "$X$",
                "*",
                "^",
                "s",
                "p",
                "1",
                "8",
                "D",
                "v",
                "<",
                ">",
                "h",
                "H",
                "+",
                "x",
            ]
            markers = [
                default_markers[i % len(default_markers)]
                for i in range(len(categories))
            ]
            cat_to_marker = dict(zip(categories, markers))
            return {
                "legend": True,
                "array": col.map(cat_to_marker),
                "categories": categories,
                "cat_to_marker": cat_to_marker,
                "col": col,
                "col_name": value,
            }

        # Fixed marker string or None → default "o", no legend
        fixed = value if isinstance(value, str) else "o"
        return {
            "legend": False,
            "array": pd.Series([fixed] * len(df)),
        }

    else:
        raise ValueError(
            f'Unknown aesthetic kind: "{kind}". Use "color", "size", "alpha", or "shape".'
        )


def _build_legend_handles(kind: str, meta: dict[str, Any], fallback_color: str = "steelblue") -> list:
    """HELPER
    Creates matplotlib artist objects to serve as legend handles for an aesthetic.

    Parameters
    ----------
    kind : {"color", "size", "alpha", "shape"}
        The aesthetic type to generate handles for.
    meta : dict
        Metadata dictionary returned by `_resolve_aesthetic`.
    fallback_color : str, default "steelblue"
        A representative color used for legends not based on color mapping.

    Returns
    -------
    list
        A list of matplotlib Artist objects (Patches or Lines).
    """

    # ── COLOR ──────────────────────────────────────────────────────────────────
    if kind == "color":
        return [
            Patch(
                facecolor=meta["hex_colors"][i],
                edgecolor="black",
                label=str(cat),
            )
            for i, cat in enumerate(meta["categories"])
        ]

    # ── SIZE ───────────────────────────────────────────────────────────────────
    elif kind == "size":
        s_arr = meta["array"]
        p01 = s_arr.quantile(0.01)
        p50 = s_arr.quantile(0.50)
        p99 = s_arr.quantile(0.99)
        raw = meta["raw_col"]
        labels = [
            f"{raw.quantile(0.01):.1f}",
            f"{raw.quantile(0.50):.1f}",
            f"{raw.quantile(0.99):.1f}",
        ]
        return [
            Line2D(
                [0],
                [0],
                marker="o",
                color=fallback_color,
                markerfacecolor=fallback_color,
                markersize=np.sqrt(sz),  # scatter uses area → sqrt for radius
                linestyle="",
                label=lbl,
            )
            for sz, lbl in zip([p01, p50, p99], labels)
        ]

    # ── ALPHA ──────────────────────────────────────────────────────────────────
    elif kind == "alpha":
        a_arr = meta["array"]
        levels = [
            a_arr.min(),
            (a_arr.min() + a_arr.max()) / 2,
            a_arr.max(),
        ]
        labels = ["Low", "Mid", "High"]
        return [
            Line2D(
                [0],
                [0],
                marker="o",
                color=fallback_color,
                markerfacecolor=fallback_color,
                markersize=10,
                alpha=a,
                linestyle="",
                label=lbl,
            )
            for a, lbl in zip(levels, labels)
        ]

    # ── SHAPE ──────────────────────────────────────────────────────────────────
    elif kind == "shape":
        return [
            Line2D(
                [0],
                [0],
                marker=marker,
                color=fallback_color,
                markerfacecolor=fallback_color,
                markersize=10,
                linestyle="",
                label=str(cat),
            )
            for cat, marker in meta["cat_to_marker"].items()
        ]

    else:
        return []


def _place_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    handles: list,
    title: str,
    x_anchor: float,
    y_anchor: float,
    gap: float = 0.02,
    **kwargs: Any
) -> float:
    """HELPER
    Places a legend box on the figure and calculates the vertical coordinate 
    for the next potential legend.

    Parameters
    ----------
    fig : plt.Figure
        The figure object where the legend will be added.
    ax : plt.Axes
        The axes object used for coordinate transformations.
    handles : list
        List of matplotlib artist objects for the legend entries.
    title : str
        The title displayed at the top of the legend box.
    x_anchor : float
        Horizontal position in axes-fraction units.
    y_anchor : float
        Vertical position (top edge) in axes-fraction units.
    gap : float, default 0.02
        Vertical spacing to subtract from the legend bottom for the next anchor.

    Returns
    -------
    float
        The y-coordinate for the top of the next legend box.
    """

    legend = fig.legend(
        handles=handles,
        title=title,
        bbox_to_anchor=(x_anchor, y_anchor),
        loc="upper left",
        prop={"size": 11},
        bbox_transform=ax.transAxes,
        framealpha=0.85,
        **kwargs,
    )
    ax.add_artist(legend)

    # Convert the legend bounding box from display coords → axes-fraction coords
    # so we know exactly where it ends vertically.
    fig.canvas.draw()
    bbox_display = legend.get_window_extent()
    bbox_axes = bbox_display.transformed(ax.transAxes.inverted())
    next_y = bbox_axes.y0 - gap  # bottom of legend minus gap

    return next_y


def _draw_fitline(ax: plt.Axes, fig: plt.Figure, x: np.ndarray, y: np.ndarray, mode: str) -> None:
    """HELPER
    Overlays regression fit lines and a corresponding legend onto a scatter plot.

    Parameters
    ----------
    ax : plt.Axes
        The axes to draw the regression lines on.
    fig : plt.Figure
        The figure object to which the fit-line legend is added.
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.
    mode : {"linear", "quadratic", "both"}
        The type of regression model to visualize.

    Returns
    -------
    None
    """
    
    x_fit = np.linspace(x.min(), x.max(), 300)
    legend_elements = []

    # ── LINEAR fit ─────────────────────────────────────────────────────────────
    if mode in ("linear", "both"):
        slope, intercept, *_ = sp.stats.linregress(x, y)
        y_fit = intercept + slope * x_fit
        ax.plot(
            x_fit,
            y_fit,
            color="darkgreen",
            linestyle="-",
            linewidth=2.5,
            zorder=5,
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="darkgreen",
                linestyle="-",
                linewidth=2.5,
                label="Linear fit",
            )
        )

    # ── QUADRATIC fit ──────────────────────────────────────────────────────────
    if mode in ("quadratic", "both"):

        def _quad(x, a, b, c):
            return a * x**2 + b * x + c

        popt, _ = sp.optimize.curve_fit(_quad, x, y, maxfev=10_000)
        y_fit = _quad(x_fit, *popt)
        ax.plot(
            x_fit,
            y_fit,
            color="mediumpurple",
            linestyle="--",
            linewidth=2.5,
            zorder=5,
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="mediumpurple",
                linestyle="--",
                linewidth=2.5,
                label="Quadratic fit",
            )
        )

    # ── Fitline legend (bottom-right corner) ───────────────────────────────────
    if legend_elements:
        fit_legend = fig.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.0, 0.0),
            loc="lower right",
            prop={"size": 11},
            bbox_transform=ax.transAxes,
            framealpha=0.85,
        )
        ax.add_artist(fit_legend)


def scatter_2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | Any | None = None,
    size: str | float | int | None = None,
    alpha: str | float | int | None = None,
    shape: str | None = None,
    fitline: str | None = None,
    edgecolor: str = "none",
    facealpha: float | int | None = None,
    edgewidth: float | int = 1,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generates a 2D scatter plot with multi-dimensional aesthetic mappings and 
    optional regression lines.

    Parameters
    ----------
    df : pd.DataFrame
        The source dataset.
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.
    color : str, scalar, or None, default None
        Column name for color mapping, or a fixed matplotlib color string.
    size : str, scalar, or None, default None
        Column name for size mapping, or a fixed marker area.
    alpha : str, scalar, or None, default None
        Column name for transparency mapping, or a fixed alpha value [0, 1].
    shape : str or None, default None
        Column name for marker shape mapping, or a fixed marker symbol.
    fitline : {"linear", "quadratic", "both"} or None, default None
        Option to overlay regression fit lines.
    edgecolor : str, default "none"
        The color of the marker borders.
    facealpha : float or None, default None
        Overrides marker face transparency independently of edge transparency.
    edgewidth : float or int, default 1
        The width of the marker borders.

    Returns
    -------
    fig : plt.Figure
        The resulting figure object.
    ax : plt.Axes
        The resulting axes object.
    """
    
    # ── 0. Input validation ────────────────────────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f'"df" must be a pandas DataFrame, got {type(df).__name__}.'
        )
    for col_arg, col_name in [(x, "x"), (y, "y")]:
        if col_arg not in df.columns:
            raise ValueError(
                f'Column "{col_arg}" (passed as "{col_name}") not found in df.'
            )
    if fitline not in (
        "linear",
        "quadratic",
        "both",
        None,
    ):
        raise ValueError(
            '"fitline" must be "linear", "quadratic", "both", or None.'
        )

    np.random.seed(42)

    x_data = df[x]
    y_data = df[y]

    # ── 1. RESOLVE aesthetics ──────────────────────────────────────────────────
    # Each call returns a dict with at minimum {"legend": bool, "array": Series}
    # plus extra metadata when legend=True (categories, hex_colors, etc.)
    c_meta = _resolve_aesthetic(color, df, "color")
    s_meta = _resolve_aesthetic(size, df, "size")
    a_meta = _resolve_aesthetic(alpha, df, "alpha")
    m_meta = _resolve_aesthetic(shape, df, "shape")

    # Fallback color for non-color legend handles (size, alpha, shape use this)
    # Use the first fixed color if color is not column-mapped
    fallback_color = "steelblue" if not c_meta["legend"] else "steelblue"

    # ── 1b. BAKE facealpha into facecolor RGBA tuples (if requested) ───────────
    # When facealpha is set, we convert every facecolor to an explicit RGBA tuple
    # with the requested alpha in the A channel. This decouples face transparency
    # from edge transparency, because matplotlib's `alpha=` kwarg hits both equally.
    # By embedding alpha into the color itself and NOT passing `alpha=` to scatter,
    # only the face is affected. edgecolor retains its own full opacity.

    if facealpha is not None:
        if c_meta["legend"]:
            # Column-mapped colors: hex_colors list → RGBA tuples with facealpha
            rgba_per_category = [
                to_rgba(h, alpha=facealpha) for h in c_meta["hex_colors"]
            ]
            # Rebuild array as per-point RGBA tuples (no cmap needed anymore)
            cat_to_rgba = dict(zip(c_meta["categories"], rgba_per_category))
            c_meta["array"] = df[color].map(cat_to_rgba)
            c_meta["cmap"] = (
                None  # cmap is for floats; we now have explicit RGBA
            )
        else:
            # Fixed color: single RGBA tuple repeated for every point
            fixed_color = c_meta["array"].iloc[0]  # e.g. "steelblue"
            rgba = to_rgba(fixed_color, alpha=facealpha)
            c_meta["array"] = pd.Series([rgba] * len(df))
            c_meta["cmap"] = None

    # ── 2. PLOT ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))

    # Shared kwargs passed to every ax.scatter() call
    scatter_kwargs = dict(
        cmap=c_meta.get("cmap"),
        edgecolor=edgecolor,
        linewidths=edgewidth,
    )

    # Hard-code alpha logic into a single variable
    alpha = (
        None
        if facealpha is not None
        else (
            a_meta["array"].mean()
            if a_meta["legend"]
            else a_meta["array"].iloc[0]
        )
    )

    # Call the ax.scatter() functions
    if m_meta["legend"]:
        # When shape is column-mapped we must call ax.scatter() once per marker
        # symbol, because matplotlib does not support mixed markers in one call.
        for category in m_meta["categories"]:
            mask = m_meta["col"] == category
            marker = m_meta["cat_to_marker"][category]
            ax.scatter(
                x_data[mask],
                y_data[mask],
                c=c_meta["array"][mask],
                s=s_meta["array"][mask],
                alpha=alpha,
                marker=marker,
                **scatter_kwargs,
            )
    else:
        # Single ax.scatter() call — fastest path
        ax.scatter(
            x_data,
            y_data,
            c=c_meta["array"],
            s=s_meta["array"],
            alpha=alpha,
            marker=m_meta["array"].iloc[0],
            **scatter_kwargs,
        )

    # ── 3. LEGENDS ─────────────────────────────────────────────────────────────
    # Stack legend boxes top-to-bottom on the right side of the plot.
    # x_anchor = 1.0 → immediately right of the axes.
    # y_anchor starts at 1.0 (top) and is decremented after each box.

    x_anchor = 1.02
    y_anchor = 1.0

    # Ordered list: which aesthetics to legend, in display order
    aesthetics_to_legend = [
        ("color", c_meta, {}),
        (
            "size",
            s_meta,
            {"labelspacing": 1.8},
        ),  # extra space between size bubbles
        ("alpha", a_meta, {}),
        ("shape", m_meta, {}),
    ]

    for (
        kind,
        meta,
        extra_kwargs,
    ) in aesthetics_to_legend:
        if not meta["legend"]:
            continue

        handles = _build_legend_handles(kind, meta, fallback_color)
        title = meta["col_name"]
        y_anchor = _place_legend(
            fig,
            ax,
            handles,
            title,
            x_anchor,
            y_anchor,
            gap=0.02,
            **extra_kwargs,
        )

    # ── 4. FITLINES ────────────────────────────────────────────────────────────
    if fitline is not None:
        _draw_fitline(
            ax,
            fig,
            x_data.values.astype(float),
            y_data.values.astype(float),
            fitline,
        )

    # ── 5. LABELS & LAYOUT ─────────────────────────────────────────────────────
    ax.set_title(
        f'"{y}" ~ "{x}"',
        fontsize=15,
        pad=18,
        fontweight="semibold",
    )
    ax.set_xlabel(f"{x}", fontsize=12)
    ax.set_ylabel(f"{y}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Make room on the right for the legends without cutting them off
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)

    plt.show()
    return fig, ax
