# =============================================================================
# MODELING MODULE
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import random as rnd
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rnd.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Function "stratified_kfold_cv()" with helpers.
# -----------------------------------------------------------------------------

def _make_stratified_folds(y: np.ndarray, k: int, n_bins: int, seed: int) -> list[np.ndarray]:
    """HELPER
    Partitions dataset indices into K folds while maintaining the distribution 
    of the target variable through quantile binning.

    Parameters
    ----------
    y : np.ndarray
        The target variable vector used for stratification.
    k : int
        The number of folds to create.
    n_bins : int
        The number of quantile buckets used to group the target values.
    seed : int
        Random seed for shuffling reproducibility.

    Returns
    -------
    list of np.ndarray
        A list of K arrays, where each array contains the test indices for a 
        specific fold.
    """

    np.random.seed(seed)

    # ── Bin y into quantile buckets ────────────────────────────────────────────
    # pd.qcut divides y so that each bucket has (approximately) equal count.
    # If duplicate bin edges exist (many identical prices), fall back to
    # rank-based qcut which handles ties gracefully.

    # Create labels for the buckets (0 to n_bins-1)
    labels = list(range(n_bins))

    # Divide y into quantile buckets.
    try:
        buckets = pd.qcut(y, q=n_bins, labels=labels)
    except ValueError:
        buckets = pd.qcut(y.rank(method="first"), q=n_bins, labels=labels)

    buckets = np.array(buckets)

    # ── Initialize K empty fold index lists ───────────────────────────────────
    folds = [[] for _ in range(k)]

    # ── For each bucket, distribute its indices evenly across the K folds ─────
    # This loop splits each quantile bucket into K parts, then assigns
    # each part to the corresponding fold together with the corresponding parts
    # from other buckets.
    for bucket_id in labels:
        # Find the row indices that belong to the current bucket.
        bucket_indices = np.where(buckets == bucket_id)[0]

        # Shuffle the row indices within bucket to avoid ordering bias
        np.random.shuffle(bucket_indices)

        # Split the current bucket indices into K equal parts.
        chunks = np.array_split(bucket_indices, k)

        # Assign each chunk to the corresponding fold in the general folds list.
        # This creates a 2D list, every sub-list contains the row indices
        # for that fold.
        for fold_id, chunk in enumerate(chunks):
            folds[fold_id].extend(chunk)

    # Re-order each fold.
    return [np.array(sorted(fold)) for fold in folds]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """HELPER
    Calculates standard regression performance metrics between true and 
    predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        The ground truth target values.
    y_pred : np.ndarray
        The values predicted by the model.

    Returns
    -------
    dict
        A dictionary containing RMSE, MAE, R², and MAPE (Mean Absolute 
        Percentage Error).
    """

    n = len(y_true)
    errors = y_true - y_pred
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    rmse = np.sqrt(ss_res / n)
    mae = np.mean(np.abs(errors))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE: skip observations where y_true == 0 to avoid division by zero
    nonzero_mask = y_true != 0
    mape = (
        np.mean(np.abs(errors[nonzero_mask] / y_true[nonzero_mask])) * 100
        if nonzero_mask.sum() > 0
        else np.nan
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
    }


def stratified_kfold_cv(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model_fn: Callable,
    k: int = 10,
    n_bins: int = 5,
    seed: int = 42,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """MAIN
    Executes a stratified K-Fold cross-validation routine for any regression 
    model function.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The feature matrix.
    y : pd.Series or np.ndarray
        The target vector.
    model_fn : Callable
        A function with signature (X_train, y_train, X_test) that returns 
        (y_pred, logs).
    k : int, default 10
        Number of cross-validation folds.
    n_bins : int, default 5
        Number of bins used for target stratification.
    seed : int, default 42
        Seed for shuffling to ensure reproducible folds.
    verbose : bool, default True
        If True, prints progress and per-fold metrics to the console.

    Returns
    -------
    summary_df : pd.DataFrame
        A table containing metrics for every fold plus mean and standard 
        deviation summaries.
    model_logs : dict
        A collection of internal logs or training history returned by the 
        model function for each fold.
    """

    # ── 0. Input validation & conversion ──────────────────────────────────────
    if not callable(model_fn):
        raise TypeError(
            '"model_fn" must be a callable with signature (X_train, y_train, X_test) -> y_pred.'
        )
    if k < 2:
        raise ValueError(f'"k" must be at least 2, got {k}.')
    if n_bins < 2:
        raise ValueError(f'"n_bins" must be at least 2, got {n_bins}.')

    # Convert to numpy arrays — works whether inputs are DataFrames or arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of rows. Got {len(X)} and {len(y)}."
        )

    # ── 1. Build stratified folds ──────────────────────────────────────────────
    folds = _make_stratified_folds(y, k, n_bins, seed)

    # Find a list of all row indices in the df.
    all_indices = np.arange(len(y))

    # ── 2. CV loop ─────────────────────────────────────────────────────────────
    fold_results = []
    model_logs = {}

    if verbose:
        print(f"{'─' * 58}")
        print(f"  Stratified {k}-Fold CV  |  bins={n_bins}  |  n={len(y)}")
        print(f"{'─' * 58}")
        print(f"  {'Fold':<7}{'RMSE':>13}{'MAE':>13}{'R2':>10}{'MAPE':>11}")
        print(f"{'─' * 58}")

    for fold_id, test_idx in enumerate(folds):
        # Find the train row indices for the current fold, which are all the
        # row indices that are NOT in the test row indices for this fold.
        train_idx = np.setdiff1d(all_indices, test_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # ── Call the model, for CV the model is a blackbox ─────────────────────
        y_pred, fold_model_logs = model_fn(X_train, y_train, X_test)
        y_pred = np.array(y_pred, dtype=float)

        # ── Compute metrics for this fold ──────────────────────────────────────
        metrics = _compute_metrics(y_test, y_pred)
        metrics["Fold"] = fold_id + 1
        fold_results.append(metrics)

        if verbose:
            print(
                f"  {fold_id + 1:<7}"
                f"{metrics['RMSE']:>13.2f}"
                f"{metrics['MAE']:>13.2f}"
                f"{metrics['R2']:>10.4f}"
                f"{metrics['MAPE']:>10.2f}%"
            )

        # ── Append model logs if any ─────────────────────────────────────────
        model_logs[f" Fold: {fold_id + 1}"] = fold_model_logs

    # ── 3. Aggregate results ───────────────────────────────────────────────────
    results_df = pd.DataFrame(fold_results)[
        ["Fold", "RMSE", "MAE", "R2", "MAPE"]
    ]

    metric_cols = ["RMSE", "MAE", "R2", "MAPE"]

    mean_row = {col: results_df[col].mean() for col in metric_cols}
    std_row = {col: results_df[col].std() for col in metric_cols}

    mean_row["Fold"] = "Mean"
    std_row["Fold"] = "Std"

    summary_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True,
    )

    # ── 4. Print summary ───────────────────────────────────────────────────────
    if verbose:
        print(f"{'─' * 58}")
        print(
            f"  {'Mean':<7}"
            f"{mean_row['RMSE']:>13.2f}"
            f"{mean_row['MAE']:>13.2f}"
            f"{mean_row['R2']:>10.4f}"
            f"{mean_row['MAPE']:>10.2f}%"
        )
        print(
            f"  {'Std':<7}"
            f"{std_row['RMSE']:>13.2f}"
            f"{std_row['MAE']:13.2f}"
            f"{std_row['R2']:>10.4f}"
            f"{std_row['MAPE']:>10.2f}%"
        )
        print(f"{'─' * 58}\n")

    return summary_df, model_logs

# -----------------------------------------------------------------------------
# Prediction Models
# -----------------------------------------------------------------------------

def ols_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, None]:
    """
    Implements a basic Ordinary Least Squares (OLS) regression using the 
    normal equation.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.
    X_test : np.ndarray
        Test feature matrix.

    Returns
    -------
    y_pred : np.ndarray
        Predicted values for the test set.
    None
        Placeholder for consistency with the CV model contract.
    """
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    beta = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)

    return (X_test @ beta), None


def make_ridge_model(lambda_: float) -> Callable:
    """
    Factory function that creates a Ridge Regression model with a specified 
    regularization strength.

    Parameters
    ----------
    lambda_ : float
        The regularization penalty parameter (L2).

    Returns
    -------
    Callable
        A model function ready for use in cross-validation.
    """
    def ridge_model(X_train, y_train, X_test):
        X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        n = X_train.shape[1]
        beta = np.linalg.inv(X_train.T @ X_train + lambda_ * np.eye(n)) @ (
            X_train.T @ y_train
        )

        return (X_test @ beta), None

    return ridge_model


def make_gradient_descent_linreg_model(
    alpha: float = 0.001, 
    max_iter: int = 5000, 
    tolerance: float = 1e-6, 
    report_every: int = None
) -> Callable:
    """
    Factory function that creates a Linear Regression model trained via 
    Gradient Descent with feature scaling.

    Parameters
    ----------
    alpha : float, default 0.001
        The learning rate for weight updates.
    max_iter : int, default 5000
        The maximum number of training iterations.
    tolerance : float, default 1e-6
        The loss improvement threshold for early stopping.
    report_every : int, optional
        Frequency (in iterations) to log the training loss.

    Returns
    -------
    Callable
        A model function that performs scaling, training, and prediction.
    """
    def gradient_descent_linreg_model(X_train, y_train, X_test):

        # ── Preparation ────────────────────────────────────────────────────────
        n_samples, n_features = X_train.shape

        # Add intercept term to X.
        X_b_train = np.hstack([np.ones((n_samples, 1)), X_train])
        X_b_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

        # Scaling: this is really needed, if you don't believe try comment it out.
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_b_train = x_scaler.fit_transform(X_b_train)
        X_b_test = x_scaler.transform(X_b_test)
        y_train = y_train.reshape(-1, 1)
        y_train = y_scaler.fit_transform(y_train).flatten()

        # ── Initialization ─────────────────────────────────────────────────────
        w = np.zeros(n_features + 1)  # Weights initialized at 0.
        previous_loss = float("inf")
        model_logs = {"Training Loop": None}
        training_loop_logs = pd.DataFrame()

        # ── Training loop ──────────────────────────────────────────────────────
        for i in range(max_iter):
            # Forward pass: compute predictions.
            y_pred = X_b_train @ w

            # Loss computation: MSE.
            errors = y_pred - y_train
            # Divide by 2 to simplify gradient expression.
            current_loss = (1 / (2 * n_samples)) * np.sum(errors**2)

            # Convergence check: stop option.
            if abs(previous_loss - current_loss) < tolerance:
                model_logs["Early Convergence"] = {
                    "Iter": (i + 1),
                    "Loss": current_loss,
                }
                break
            previous_loss = current_loss

            # Backward pass: compute gradients.
            gradients = (1 / n_samples) * (X_b_train.T @ errors)

            # Update weights.
            w = w - (alpha * gradients)

            # Report loss if specified.
            if report_every is not None and (i + 1) % report_every == 0:
                logs_row = pd.DataFrame({"Iter": [i], "Loss": [current_loss]})
                training_loop_logs = pd.concat(
                    [training_loop_logs, logs_row], ignore_index=True
                )

        # ── Predictions ──────────────────────────────────────────────────────
        y_pred = X_b_test @ w
        y_pred = y_pred.reshape(-1, 1)
        y_pred = y_scaler.inverse_transform(y_pred).flatten()

        # ── Update logs ──────────────────────────────────────────────────────
        model_logs["Training Loop"] = training_loop_logs

        return y_pred, model_logs

    return gradient_descent_linreg_model
