"""
Results aggregation and processing utilities.

Provides functions for building and processing result DataFrames.
"""

import pandas as pd


def build_lopo_results_df(slice_df):
    """
    Aggregates results, ensuring 'n_positive' and 'n_negative' are summed correctly.

    This function groups slice-level results by fold or patient and aggregates metrics.
    Count columns (n_positive, n_negative) are summed, while performance metrics are averaged.

    Args:
        slice_df: DataFrame with slice-level results

    Returns:
        pd.DataFrame: Aggregated results grouped by fold or patient
    """
    if slice_df.empty:
        return pd.DataFrame()

    # Check for required columns
    required_cols = ['auc', 'accuracy', 'sensitivity', 'specificity', 'n_positive', 'n_negative']
    missing_cols = [col for col in required_cols if col not in slice_df.columns]

    if missing_cols:
        print(f"Warning: Missing columns in slice_df: {missing_cols}")
        # Add default values for missing columns
        for col in missing_cols:
            if col in ['n_positive', 'n_negative']:
                slice_df[col] = 0  # Default count columns to 0
            else:
                slice_df[col] = 0.0  # Default metric columns to 0.0

    # Use mean aggregation for base metrics
    metric_aggs = {
        'auc': 'mean',
        'accuracy': 'mean',
        'sensitivity': 'mean',
        'specificity': 'mean'
    }

    # Use sum aggregation for count columns (this is the key point)
    count_aggs = {
        'n_positive': 'sum',
        'n_negative': 'sum'
    }

    # Combine aggregation dictionaries
    agg_dict = {**metric_aggs, **count_aggs}

    # Add f1_score if it exists
    if 'f1_score' in slice_df.columns:
        agg_dict['f1_score'] = 'mean'

    # Group by fold and aggregate
    if 'fold' in slice_df.columns:
        lopo_results_df = slice_df.groupby('fold').agg(agg_dict).reset_index()
    elif 'patient_id' in slice_df.columns:
        lopo_results_df = slice_df.groupby('patient_id').agg(agg_dict).reset_index()
    else:
        # If no grouping column, return overall statistics
        lopo_results_df = pd.DataFrame([slice_df.agg(agg_dict)])
        lopo_results_df['fold'] = 'overall'

    # Ensure count columns are integer type
    for count_col in ['n_positive', 'n_negative']:
        if count_col in lopo_results_df.columns:
            lopo_results_df[count_col] = lopo_results_df[count_col].fillna(0).astype(int)

    return lopo_results_df
